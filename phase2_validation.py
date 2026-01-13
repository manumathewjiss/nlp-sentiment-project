"""
Phase 2: Toxicity & Hate Speech Categorization
Categorizes manually collected negative comments into toxicity categories

This script:
1. Loads the 100 manually collected negative comments
2. Categorizes them using BART zero-shot classification
3. Generates distribution analysis and visualizations
4. Creates detailed reports
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class Config:
    INPUT_CSV = "outputs/phase1_validation_results.csv"  # Phase 1 validation results
    OUTPUT_DIR = "outputs"
    CLASSIFICATION_RESULTS = "phase2_validation_classification_results.csv"
    CLASSIFICATION_REPORT = "phase2_validation_classification_report.txt"
    CATEGORY_DISTRIBUTION_IMG = "phase2_validation_category_distribution.png"
    CLASSIFICATION_MATRIX_IMG = "phase2_validation_classification_matrix.png"
    CONFIDENCE_DISTRIBUTION_IMG = "phase2_validation_confidence_distribution.png"
    MODEL_NAME = "facebook/bart-large-mnli"
    BATCH_SIZE = 1  # Zero-shot classification processes one at a time
    USE_ALL_COMMENTS = True  # If False, only uses comments predicted as negative in Phase 1


# Define the 7 classification categories
CATEGORIES = [
    "harassment or hate speech",
    "spam",
    "inappropriate content",
    "toxicity",
    "aggressive behavior",
    "misinformation",
    "other negative"
]

CATEGORY_LABELS = [
    "Harassment/Hate Speech",
    "Spam",
    "Inappropriate Content",
    "Toxicity",
    "Aggressive Behavior",
    "Misinformation",
    "Other Negative"
]


def load_comments():
    """Load comments from Phase 1 validation results"""
    print("="*80)
    print("PHASE 2: TOXICITY & HATE SPEECH CATEGORIZATION")
    print("Categorizing Manually Collected Negative Comments")
    print("="*80)
    
    print("\n" + "="*80)
    print("STEP 1: LOADING COMMENTS")
    print("="*80)
    
    if not os.path.exists(Config.INPUT_CSV):
        raise FileNotFoundError(f"File not found: {Config.INPUT_CSV}")
    
    print(f"📂 Reading: {Config.INPUT_CSV}")
    df = pd.read_csv(Config.INPUT_CSV)
    
    print(f"✅ Loaded {len(df)} comments from Phase 1 validation")
    
    # Filter comments based on configuration
    if Config.USE_ALL_COMMENTS:
        print(f"📊 Using ALL {len(df)} comments for categorization")
        filtered_df = df.copy()
    else:
        # Only use comments that were correctly identified as negative in Phase 1
        filtered_df = df[df['Predicted_Sentiment'] == 'negative'].copy()
        print(f"📊 Using {len(filtered_df)} comments predicted as NEGATIVE in Phase 1")
        print(f"   (Filtered out {len(df) - len(filtered_df)} comments)")
    
    print(f"\nDataFrame columns: {list(filtered_df.columns)}")
    
    # Ensure we have clean_text column
    if 'clean_text' not in filtered_df.columns:
        if 'Comment_Text' in filtered_df.columns:
            from preprocess import clean_text
            filtered_df['clean_text'] = filtered_df['Comment_Text'].apply(clean_text)
            print("✅ Created clean_text column from Comment_Text")
        else:
            raise ValueError("No 'clean_text' or 'Comment_Text' column found")
    
    print(f"\n📝 Sample comments (first 3):")
    for i in range(min(3, len(filtered_df))):
        comment = filtered_df.iloc[i]['clean_text'][:100]
        collection_num = filtered_df.iloc[i].get('Collection_Number', i+1)
        print(f"   {collection_num}. {comment}...")
    
    return filtered_df


def load_classifier():
    """Load the BART zero-shot classification model"""
    print("\n" + "="*80)
    print("STEP 2: LOADING BART ZERO-SHOT CLASSIFICATION MODEL")
    print("="*80)
    
    print("🤖 Model: facebook/bart-large-mnli (Zero-Shot Classification)")
    print("📋 Categories:")
    for i, category in enumerate(CATEGORY_LABELS, 1):
        print(f"   {i}. {category}")
    print("\n⏳ Loading model... (This may take a few minutes on first run)")
    
    classifier = pipeline(
        "zero-shot-classification",
        model=Config.MODEL_NAME,
        device=-1  # Use CPU (-1), change to 0 for GPU
    )
    
    print("✅ Model loaded successfully!")
    return classifier


def classify_comments(df, classifier):
    """Classify all comments into toxicity categories"""
    print("\n" + "="*80)
    print("STEP 3: CLASSIFYING COMMENTS")
    print("="*80)
    
    print(f"🔄 Processing {len(df)} comments...")
    print("⏳ This will take some time. Progress will be shown every 10 comments.\n")
    
    predictions = []
    confidence_scores = []
    all_scores = []  # Store all category scores for analysis
    
    for idx, row in df.iterrows():
        comment = row['clean_text']
        
        try:
            # Perform zero-shot classification
            result = classifier(
                comment,
                candidate_labels=CATEGORIES,
                multi_label=False  # Single label classification
            )
            
            # Get the top prediction
            predicted_category = result['labels'][0]
            confidence = result['scores'][0]
            
            # Map to readable label
            category_idx = CATEGORIES.index(predicted_category)
            predicted_label = CATEGORY_LABELS[category_idx]
            
            predictions.append(predicted_label)
            confidence_scores.append(confidence)
            all_scores.append(result['scores'])  # Store all scores
            
            # Show progress
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(df)} comments ({(idx+1)/len(df)*100:.1f}%)")
        
        except Exception as e:
            print(f"   ⚠️  Error processing comment {idx+1}: {str(e)}")
            predictions.append("Unknown")
            confidence_scores.append(0.0)
            all_scores.append([0.0] * len(CATEGORIES))
    
    df['Predicted_Category'] = predictions
    df['Confidence_Score'] = confidence_scores
    
    print(f"\n✅ Classification complete! Processed {len(df)} comments")
    
    return df, all_scores


def analyze_results(df):
    """Analyze classification results"""
    print("\n" + "="*80)
    print("STEP 4: ANALYZING RESULTS")
    print("="*80)
    
    category_counts = Counter(df['Predicted_Category'])
    
    print("\n📊 Category Distribution:")
    for category in CATEGORY_LABELS:
        count = category_counts.get(category, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"   {category:30s}: {count:5d} ({percentage:5.2f}%)")
    
    print(f"\n📈 Confidence Score Statistics:")
    print(f"   Average: {np.mean(df['Confidence_Score']):.2%}")
    print(f"   Median:  {np.median(df['Confidence_Score']):.2%}")
    print(f"   Min:     {np.min(df['Confidence_Score']):.2%}")
    print(f"   Max:     {np.max(df['Confidence_Score']):.2%}")
    print(f"   Std Dev: {np.std(df['Confidence_Score']):.2%}")
    
    return category_counts


def create_visualizations(df, category_counts):
    """Create visualization charts"""
    print("\n" + "="*80)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Category Distribution Bar Chart
    print("\n📊 Creating category distribution chart...")
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    percentages = [(c / len(df)) * 100 for c in counts]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(categories)), counts, color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Toxicity Category', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Comments', fontsize=13, fontweight='bold')
    plt.title('Phase 2 Validation: Distribution of Toxicity Categories\n100 Manually Collected Negative Comments', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    dist_path = os.path.join(Config.OUTPUT_DIR, Config.CATEGORY_DISTRIBUTION_IMG)
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {dist_path}")
    plt.close()
    
    # 2. Classification Matrix (showing distribution)
    print("\n📊 Creating classification matrix...")
    category_matrix = np.zeros((len(CATEGORY_LABELS), len(CATEGORY_LABELS)))
    for i, cat in enumerate(CATEGORY_LABELS):
        category_matrix[i][i] = category_counts.get(cat, 0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        category_matrix,
        annot=True,
        fmt='g',
        cmap='YlOrRd',
        xticklabels=CATEGORY_LABELS,
        yticklabels=CATEGORY_LABELS,
        cbar_kws={'label': 'Count'},
        square=True
    )
    plt.title('Phase 2 Validation: Toxicity Classification Distribution\nBART Zero-Shot Model', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
    plt.ylabel('Category', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    matrix_path = os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_MATRIX_IMG)
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {matrix_path}")
    plt.close()
    
    # 3. Confidence Score Distribution
    print("\n📊 Creating confidence score distribution...")
    plt.figure(figsize=(12, 6))
    plt.hist(df['Confidence_Score'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(df['Confidence_Score']), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(df["Confidence_Score"]):.2%}')
    plt.axvline(np.median(df['Confidence_Score']), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(df["Confidence_Score"]):.2%}')
    plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Phase 2 Validation: Model Confidence Score Distribution\nBART Zero-Shot Classification', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    conf_path = os.path.join(Config.OUTPUT_DIR, Config.CONFIDENCE_DISTRIBUTION_IMG)
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {conf_path}")
    plt.close()


def save_results(df, category_counts):
    """Save classification results"""
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Save full classification results
    results_path = os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_RESULTS)
    output_cols = [
        'Collection_Number', 'Comment_Text', 'clean_text',
        'Predicted_Category', 'Confidence_Score',
        'Username', 'Channel_ID', 'Video_URL',
        'Ground_Truth_Sentiment', 'Predicted_Sentiment'  # From Phase 1
    ]
    available_cols = [col for col in output_cols if col in df.columns]
    df[available_cols].to_csv(results_path, index=False)
    print(f"✅ Saved: {results_path}")
    
    # Generate detailed report
    report_path = os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_REPORT)
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2 VALIDATION: TOXICITY & HATE SPEECH CATEGORIZATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Source: Manually collected negative comments\n")
        f.write(f"Total Comments Analyzed: {len(df)}\n")
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Type: BART Zero-Shot Classification\n")
        f.write(f"Classification Type: Multi-class (Single Label)\n\n")
        
        f.write("CLASSIFICATION CATEGORIES\n")
        f.write("-"*80 + "\n")
        for i, category in enumerate(CATEGORY_LABELS, 1):
            f.write(f"{i}. {category}\n")
        f.write("\n")
        
        f.write("CATEGORY DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Category':<35} {'Count':>10} {'Percentage':>12}\n")
        f.write("-"*80 + "\n")
        for category in CATEGORY_LABELS:
            count = category_counts.get(category, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            f.write(f"{category:<35} {count:>10} {percentage:>11.2f}%\n")
        f.write("-"*80 + "\n")
        f.write(f"{'TOTAL':<35} {len(df):>10} {100.00:>11.2f}%\n\n")
        
        f.write("CONFIDENCE SCORE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Confidence: {np.mean(df['Confidence_Score']):.2%}\n")
        f.write(f"Median Confidence:  {np.median(df['Confidence_Score']):.2%}\n")
        f.write(f"Std Deviation:      {np.std(df['Confidence_Score']):.2%}\n")
        f.write(f"Min Confidence:     {np.min(df['Confidence_Score']):.2%}\n")
        f.write(f"Max Confidence:     {np.max(df['Confidence_Score']):.2%}\n\n")
        
        f.write("TOP COMMENTS BY CATEGORY\n")
        f.write("-"*80 + "\n")
        for category in CATEGORY_LABELS:
            category_df = df[df['Predicted_Category'] == category]
            if len(category_df) > 0:
                f.write(f"\n{category} ({len(category_df)} comments):\n")
                top_5 = category_df.nlargest(5, 'Confidence_Score')
                for idx, row in top_5.iterrows():
                    comment = row['clean_text'][:120]
                    confidence = row['Confidence_Score']
                    f.write(f"  - [{confidence:.2%}] {comment}...\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("GENERATED FILES\n")
        f.write("="*80 + "\n")
        f.write(f"1. {results_path}\n")
        f.write(f"2. {report_path}\n")
        f.write(f"3. {os.path.join(Config.OUTPUT_DIR, Config.CATEGORY_DISTRIBUTION_IMG)}\n")
        f.write(f"4. {os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_MATRIX_IMG)}\n")
        f.write(f"5. {os.path.join(Config.OUTPUT_DIR, Config.CONFIDENCE_DISTRIBUTION_IMG)}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Saved: {report_path}")


def display_summary(df, category_counts):
    """Display validation summary"""
    print("\n" + "="*80)
    print("PHASE 2 VALIDATION COMPLETE! 🎉")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Classified {len(df)} negative comments")
    print(f"   ✅ Used BART Zero-Shot Model ({Config.MODEL_NAME})")
    print(f"   ✅ Average confidence: {np.mean(df['Confidence_Score']):.2%}")
    print(f"   ✅ Generated visualizations and detailed report")
    
    print(f"\n📈 Category Distribution:")
    for category in CATEGORY_LABELS:
        count = category_counts.get(category, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"   {category:30s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n📁 Generated Files:")
    print(f"   1. {os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_RESULTS)}")
    print(f"   2. {os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_REPORT)}")
    print(f"   3. {os.path.join(Config.OUTPUT_DIR, Config.CATEGORY_DISTRIBUTION_IMG)}")
    print(f"   4. {os.path.join(Config.OUTPUT_DIR, Config.CLASSIFICATION_MATRIX_IMG)}")
    print(f"   5. {os.path.join(Config.OUTPUT_DIR, Config.CONFIDENCE_DISTRIBUTION_IMG)}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    start_time = datetime.now()
    
    try:
        # Step 1: Load comments
        df = load_comments()
        
        # Step 2: Load classifier
        classifier = load_classifier()
        
        # Step 3: Classify comments
        df, all_scores = classify_comments(df, classifier)
        
        # Step 4: Analyze results
        category_counts = analyze_results(df)
        
        # Step 5: Create visualizations
        create_visualizations(df, category_counts)
        
        # Step 6: Save results
        save_results(df, category_counts)
        
        # Display summary
        display_summary(df, category_counts)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Total Execution Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
