"""
Phase 1 Model Validation Script
Validates RoBERTa sentiment model on manually collected 100 negative comments

This script:
1. Loads the 100 manually collected negative comments
2. Runs RoBERTa sentiment analysis
3. Validates that model correctly identifies them as negative
4. Generates validation report and visualizations
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from preprocess import clean_text
import numpy as np

class Config:
    INPUT_CSV = "../task1_collection_template.csv"  # Path to your manually collected data
    OUTPUT_DIR = "outputs"
    VALIDATION_RESULTS = "phase1_validation_results.csv"
    VALIDATION_REPORT = "phase1_validation_report.txt"
    CONFUSION_MATRIX_IMG = "phase1_validation_confusion_matrix.png"
    PREDICTION_DISTRIBUTION_IMG = "phase1_validation_prediction_distribution.png"
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    CACHE_DIR = "./model_cache"
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EXPECTED_SENTIMENT = "negative"  # All comments are manually identified as negative


def load_validation_dataset(file_path):
    """Load the manually collected validation dataset"""
    print("\n" + "="*80)
    print("STEP 1: LOADING VALIDATION DATASET")
    print("="*80)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading validation data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"✅ Successfully loaded {len(df):,} manually collected comments")
    print(f"Columns: {list(df.columns)}")
    
    # Add ground truth label (all should be negative)
    df['Ground_Truth_Sentiment'] = Config.EXPECTED_SENTIMENT
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total comments: {len(df)}")
    print(f"   Expected sentiment: {Config.EXPECTED_SENTIMENT.upper()}")
    print(f"   All comments manually identified as: NEGATIVE")
    
    print(f"\n📝 Sample comments (first 3):")
    for i in range(min(3, len(df))):
        comment = df.loc[i, 'Comment_Text'][:80]
        username = df.loc[i, 'Username']
        print(f"   {i+1}. [{username}] {comment}...")
    
    return df


def preprocess_validation_data(df):
    """Preprocess the validation comments"""
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING COMMENTS")
    print("="*80)
    
    print(f"Cleaning {len(df):,} comments...")
    print("   (Removing URLs, mentions, hashtags, special characters)")
    
    tqdm.pandas(desc="Cleaning")
    df['clean_text'] = df['Comment_Text'].progress_apply(clean_text)
    
    initial_count = len(df)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"⚠️  Removed {removed} empty comments after cleaning")
    
    print(f"✅ Preprocessing complete! {len(df):,} valid comments ready")
    
    print(f"\n📝 Before/After examples:")
    for i in range(min(2, len(df))):
        original = df.loc[i, 'Comment_Text'][:70]
        cleaned = df.loc[i, 'clean_text'][:70]
        print(f"\n   Example {i+1}:")
        print(f"     Original: {original}...")
        print(f"     Cleaned:  {cleaned}...")
    
    return df


def load_model():
    """Load the RoBERTa sentiment model"""
    print("\n" + "="*80)
    print("STEP 3: LOADING ROBERTA SENTIMENT MODEL")
    print("="*80)
    
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Cache directory: {Config.CACHE_DIR}")
    
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    print("\n⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME, 
        cache_dir=Config.CACHE_DIR
    )
    print("✅ Tokenizer loaded")
    
    print("\n⏳ Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR
    )
    print("✅ Model loaded")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get label mapping
    labels = ['negative', 'neutral', 'positive']
    print(f"\n📋 Model labels: {labels}")
    
    return tokenizer, model, labels


def predict_sentiment(df, tokenizer, model, labels):
    """Run sentiment predictions on all comments"""
    print("\n" + "="*80)
    print("STEP 4: RUNNING SENTIMENT PREDICTIONS")
    print("="*80)
    
    print(f"🔄 Processing {len(df):,} comments...")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Max length: {Config.MAX_LENGTH}")
    
    texts = df['clean_text'].tolist()
    predictions = []
    confidence_scores = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   Using device: {device}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), Config.BATCH_SIZE), desc="Predicting"):
            batch = texts[i:i+Config.BATCH_SIZE]
            
            # Tokenize
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Predict
            outputs = model(**tokens)
            probs = F.softmax(outputs.logits, dim=1)
            
            # Get predictions and confidence
            for prob in probs:
                pred_idx = prob.argmax().item()
                pred_label = labels[pred_idx]
                confidence = prob[pred_idx].item()
                
                predictions.append(pred_label)
                confidence_scores.append(confidence)
    
    df['Predicted_Sentiment'] = predictions
    df['Confidence_Score'] = confidence_scores
    
    print(f"\n✅ Predictions complete!")
    print(f"   Processed: {len(predictions):,} comments")
    
    return df


def evaluate_model(df):
    """Evaluate model performance"""
    print("\n" + "="*80)
    print("STEP 5: EVALUATING MODEL PERFORMANCE")
    print("="*80)
    
    y_true = df['Ground_Truth_Sentiment'].tolist()
    y_pred = df['Predicted_Sentiment'].tolist()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n📊 Validation Results:")
    print(f"   Total comments: {len(df)}")
    print(f"   Expected sentiment: {Config.EXPECTED_SENTIMENT.upper()}")
    print(f"   Accuracy: {accuracy:.2%}")
    
    # Count predictions
    pred_counts = pd.Series(y_pred).value_counts()
    print(f"\n📈 Prediction Distribution:")
    for sentiment, count in pred_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.upper():10s}: {count:3d} ({percentage:5.2f}%)")
    
    # Calculate per-class metrics
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    
    # Calculate additional metrics
    correct_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 'negative')
    total_negative = len([x for x in y_true if x == 'negative'])
    
    precision_negative = correct_negative / total_negative if total_negative > 0 else 0
    recall_negative = correct_negative / total_negative if total_negative > 0 else 0
    
    print(f"\n🎯 Key Metrics:")
    print(f"   Overall Accuracy: {accuracy:.2%}")
    print(f"   Negative Precision: {precision_negative:.2%}")
    print(f"   Negative Recall: {recall_negative:.2%}")
    print(f"   Correctly identified as negative: {correct_negative}/{total_negative}")
    
    metrics = {
        'accuracy': accuracy,
        'precision_negative': precision_negative,
        'recall_negative': recall_negative,
        'correct_negative': correct_negative,
        'total_negative': total_negative,
        'confusion_matrix': cm
    }
    
    return metrics


def create_visualizations(df, metrics):
    """Create visualization charts"""
    print("\n" + "="*80)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Confusion Matrix
    print("\n📊 Creating confusion matrix...")
    cm = metrics['confusion_matrix']
    labels = ['negative', 'neutral', 'positive']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Phase 1 Validation: Confusion Matrix\nRoBERTa Model on 100 Manually Collected Negative Comments', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Sentiment (All Negative)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    confusion_path = os.path.join(Config.OUTPUT_DIR, Config.CONFUSION_MATRIX_IMG)
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {confusion_path}")
    plt.close()
    
    # 2. Prediction Distribution
    print("\n📊 Creating prediction distribution chart...")
    pred_counts = df['Predicted_Sentiment'].value_counts()
    colors = {'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#2ecc71'}
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        pred_counts.index,
        pred_counts.values,
        color=[colors.get(sent, '#3498db') for sent in pred_counts.index],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Comments', fontsize=12, fontweight='bold')
    plt.title('Phase 1 Validation: Prediction Distribution\nExpected: 100% Negative | Actual Results', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(fontsize=11)
    plt.tight_layout()
    
    dist_path = os.path.join(Config.OUTPUT_DIR, Config.PREDICTION_DISTRIBUTION_IMG)
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {dist_path}")
    plt.close()
    
    # 3. Confidence Score Distribution
    print("\n📊 Creating confidence score distribution...")
    plt.figure(figsize=(12, 6))
    plt.hist(df['Confidence_Score'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(df['Confidence_Score'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df["Confidence_Score"].mean():.2%}')
    plt.axvline(df['Confidence_Score'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {df["Confidence_Score"].median():.2%}')
    plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Phase 1 Validation: Model Confidence Score Distribution', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    conf_path = os.path.join(Config.OUTPUT_DIR, "phase1_validation_confidence_distribution.png")
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {conf_path}")
    plt.close()


def save_results(df, metrics):
    """Save validation results to files"""
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Save full results CSV
    results_path = os.path.join(Config.OUTPUT_DIR, Config.VALIDATION_RESULTS)
    output_cols = [
        'Collection_Number', 'Comment_Text', 'clean_text', 
        'Ground_Truth_Sentiment', 'Predicted_Sentiment', 'Confidence_Score',
        'Username', 'Channel_ID', 'Video_URL'
    ]
    df[output_cols].to_csv(results_path, index=False)
    print(f"✅ Saved: {results_path}")
    
    # Generate detailed report
    report_path = os.path.join(Config.OUTPUT_DIR, Config.VALIDATION_REPORT)
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1 MODEL VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("VALIDATION DATASET\n")
        f.write("-"*80 + "\n")
        f.write(f"Source: Manually collected negative comments\n")
        f.write(f"Total Comments: {len(df)}\n")
        f.write(f"Expected Sentiment: {Config.EXPECTED_SENTIMENT.upper()}\n")
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Model Type: RoBERTa (Twitter Sentiment)\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"Max Length: {Config.MAX_LENGTH}\n\n")
        
        f.write("VALIDATION RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n")
        f.write(f"Negative Precision: {metrics['precision_negative']:.2%}\n")
        f.write(f"Negative Recall: {metrics['recall_negative']:.2%}\n")
        f.write(f"Correctly Identified as Negative: {metrics['correct_negative']}/{metrics['total_negative']}\n\n")
        
        f.write("PREDICTION DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        pred_counts = df['Predicted_Sentiment'].value_counts()
        for sentiment, count in pred_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment.upper():10s}: {count:3d} ({percentage:5.2f}%)\n")
        f.write("\n")
        
        f.write("CONFIDENCE SCORE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Confidence: {df['Confidence_Score'].mean():.2%}\n")
        f.write(f"Median Confidence: {df['Confidence_Score'].median():.2%}\n")
        f.write(f"Min Confidence: {df['Confidence_Score'].min():.2%}\n")
        f.write(f"Max Confidence: {df['Confidence_Score'].max():.2%}\n")
        f.write(f"Std Deviation: {df['Confidence_Score'].std():.2%}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        y_true = df['Ground_Truth_Sentiment'].tolist()
        y_pred = df['Predicted_Sentiment'].tolist()
        f.write(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))
        f.write("\n")
        
        f.write("MISCLASSIFIED COMMENTS\n")
        f.write("-"*80 + "\n")
        misclassified = df[df['Ground_Truth_Sentiment'] != df['Predicted_Sentiment']]
        if len(misclassified) > 0:
            f.write(f"Found {len(misclassified)} misclassified comments:\n\n")
            for idx, row in misclassified.iterrows():
                f.write(f"Collection #{row['Collection_Number']}:\n")
                f.write(f"  Comment: {row['Comment_Text'][:150]}...\n")
                f.write(f"  Expected: {row['Ground_Truth_Sentiment']}\n")
                f.write(f"  Predicted: {row['Predicted_Sentiment']}\n")
                f.write(f"  Confidence: {row['Confidence_Score']:.2%}\n")
                f.write(f"  Username: {row['Username']}\n\n")
        else:
            f.write("✅ All comments correctly identified as negative!\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Saved: {report_path}")


def display_summary(df, metrics, start_time):
    """Display validation summary"""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("PHASE 1 VALIDATION COMPLETE! 🎉")
    print("="*80)
    
    print(f"\n⏱️  Execution Time: {duration:.1f} seconds")
    
    print(f"\n📊 Validation Summary:")
    print(f"   Total Comments: {len(df)}")
    print(f"   Expected: All NEGATIVE")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Correctly Identified: {metrics['correct_negative']}/{metrics['total_negative']}")
    
    pred_counts = df['Predicted_Sentiment'].value_counts()
    print(f"\n📈 Prediction Breakdown:")
    for sentiment, count in pred_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.upper():10s}: {count:3d} ({percentage:5.2f}%)")
    
    print(f"\n📁 Generated Files:")
    print(f"   1. {os.path.join(Config.OUTPUT_DIR, Config.VALIDATION_RESULTS)}")
    print(f"   2. {os.path.join(Config.OUTPUT_DIR, Config.VALIDATION_REPORT)}")
    print(f"   3. {os.path.join(Config.OUTPUT_DIR, Config.CONFUSION_MATRIX_IMG)}")
    print(f"   4. {os.path.join(Config.OUTPUT_DIR, Config.PREDICTION_DISTRIBUTION_IMG)}")
    print(f"   5. {os.path.join(Config.OUTPUT_DIR, 'phase1_validation_confidence_distribution.png')}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("PHASE 1 MODEL VALIDATION")
    print("Validating RoBERTa on 100 Manually Collected Negative Comments")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Dataset: {Config.INPUT_CSV}")
    print("="*80)
    
    try:
        # Step 1: Load dataset
        df = load_validation_dataset(Config.INPUT_CSV)
        
        # Step 2: Preprocess
        df = preprocess_validation_data(df)
        
        # Step 3: Load model
        tokenizer, model, labels = load_model()
        
        # Step 4: Predict
        df = predict_sentiment(df, tokenizer, model, labels)
        
        # Step 5: Evaluate
        metrics = evaluate_model(df)
        
        # Step 6: Visualizations
        create_visualizations(df, metrics)
        
        # Step 7: Save results
        save_results(df, metrics)
        
        # Display summary
        display_summary(df, metrics, start_time)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
