"""
Phase 2: Advanced Negative Comment Classification
Using BART Zero-Shot Classification Model

Model: facebook/bart-large-mnli
Input: Negative comments from Phase 1 RoBERTa (imbalanced dataset)
Output: Multi-class classification into 7 negative categories
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 2: ADVANCED NEGATIVE COMMENT CLASSIFICATION")
print("Model: BART Zero-Shot Classification (facebook/bart-large-mnli)")
print("="*80)

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

print("\n📊 Classification Categories:")
for i, category in enumerate(CATEGORY_LABELS, 1):
    print(f"   {i}. {category}")

# Step 1: Load the negative comments
print("\n" + "="*80)
print("STEP 1: Loading Negative Comments")
print("="*80)

input_file = "outputs/phase2_input_negative_comments_roberta_imbalanced.csv"
print(f"📂 Reading: {input_file}")

df = pd.read_csv(input_file)
print(f"✅ Loaded {len(df)} negative comments")
print(f"\nDataFrame columns: {list(df.columns)}")
print(f"\nFirst 3 comments:")
for i in range(min(3, len(df))):
    comment = df.iloc[i]['clean_text'][:100]
    print(f"   {i+1}. {comment}...")

# Step 2: Initialize the DeBERTa-v3 Zero-Shot Classification Model
print("\n" + "="*80)
print("STEP 2: Loading DeBERTa-v3 Zero-Shot Model")
print("="*80)
print("🤖 Model: facebook/bart-large-mnli (Zero-Shot Classification)")
print("⏳ This may take a few minutes to download and load...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # Use CPU (-1), change to 0 for GPU
)

print("✅ Model loaded successfully!")

# Step 3: Classify all comments
print("\n" + "="*80)
print("STEP 3: Classifying Comments")
print("="*80)
print(f"🔄 Processing {len(df)} comments...")
print("⏳ This will take some time. Progress will be shown every 100 comments.\n")

predictions = []
confidence_scores = []

for idx, row in df.iterrows():
    comment = row['clean_text']
    
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
    
    # Show progress
    if (idx + 1) % 100 == 0:
        print(f"   Processed {idx + 1}/{len(df)} comments ({(idx+1)/len(df)*100:.1f}%)")

print(f"\n✅ Classification complete! Processed {len(df)} comments")

# Add predictions to dataframe
df['Predicted_Category'] = predictions
df['Confidence_Score'] = confidence_scores

# Step 4: Generate Statistics and Distribution
print("\n" + "="*80)
print("STEP 4: Analyzing Results")
print("="*80)

category_counts = Counter(predictions)
print("\n📊 Category Distribution:")
for category in CATEGORY_LABELS:
    count = category_counts[category]
    percentage = (count / len(df)) * 100
    print(f"   {category:30s}: {count:5d} ({percentage:5.2f}%)")

print(f"\n📈 Average Confidence Score: {np.mean(confidence_scores):.2%}")
print(f"📈 Median Confidence Score: {np.median(confidence_scores):.2%}")
print(f"📈 Min Confidence Score: {np.min(confidence_scores):.2%}")
print(f"📈 Max Confidence Score: {np.max(confidence_scores):.2%}")

# Step 5: Generate Confusion Matrix
print("\n" + "="*80)
print("STEP 5: Generating Confusion Matrix")
print("="*80)

# For Phase 2, since we don't have ground truth labels, we'll create a 
# self-consistency matrix by re-classifying a sample with different random seeds
# or we can show the prediction distribution matrix

# Create a simple distribution matrix showing prediction counts
print("\n📊 Creating category distribution visualization...")

# Create confusion matrix format (showing prediction distribution)
category_matrix = np.zeros((len(CATEGORY_LABELS), len(CATEGORY_LABELS)))
for i, cat in enumerate(CATEGORY_LABELS):
    category_matrix[i][i] = category_counts[cat]

# Step 6: Generate Visualizations
print("\n" + "="*80)
print("STEP 6: Creating Visualizations")
print("="*80)

# Visualization 1: Confusion Matrix (Distribution Matrix)
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
plt.title('Phase 2: Negative Comment Classification Distribution\nBART Zero-Shot Model', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
plt.ylabel('Category', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

output_matrix = "outputs/phase2_classification_matrix.png"
plt.savefig(output_matrix, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_matrix}")
plt.close()

# Visualization 2: Category Distribution Bar Chart
plt.figure(figsize=(14, 8))
categories = list(category_counts.keys())
counts = list(category_counts.values())
percentages = [(c / len(df)) * 100 for c in counts]

bars = plt.bar(range(len(categories)), counts, color='#e74c3c', alpha=0.8, edgecolor='black')

# Add value labels on bars
for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Negative Comment Category', fontsize=13, fontweight='bold')
plt.ylabel('Number of Comments', fontsize=13, fontweight='bold')
plt.title('Phase 2: Distribution of Negative Comment Categories\nBART Zero-Shot Classification', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(range(len(categories)), categories, rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

output_dist = "outputs/phase2_category_distribution.png"
plt.savefig(output_dist, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dist}")
plt.close()

# Visualization 3: Confidence Score Distribution
plt.figure(figsize=(12, 6))
plt.hist(confidence_scores, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(confidence_scores):.2%}')
plt.axvline(np.median(confidence_scores), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {np.median(confidence_scores):.2%}')
plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Phase 2: Model Confidence Score Distribution\nBART Zero-Shot Classification', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

output_confidence = "outputs/phase2_confidence_distribution.png"
plt.savefig(output_confidence, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_confidence}")
plt.close()

# Step 7: Save Results
print("\n" + "="*80)
print("STEP 7: Saving Results")
print("="*80)

# Save full classification results
output_csv = "outputs/phase2_classification_results.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

# Generate detailed report
report_file = "outputs/phase2_classification_report.txt"
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PHASE 2: ADVANCED NEGATIVE COMMENT CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("MODEL INFORMATION\n")
    f.write("-"*80 + "\n")
    f.write("Model: facebook/bart-large-mnli\n")
    f.write("Type: BART Zero-Shot Classification\n")
    f.write("Classification Type: Multi-class (Single Label)\n")
    f.write(f"Total Comments Classified: {len(df)}\n\n")
    
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
        count = category_counts[category]
        percentage = (count / len(df)) * 100
        f.write(f"{category:<35} {count:>10} {percentage:>11.2f}%\n")
    f.write("-"*80 + "\n")
    f.write(f"{'TOTAL':<35} {len(df):>10} {100.00:>11.2f}%\n\n")
    
    f.write("CONFIDENCE SCORE STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Average Confidence: {np.mean(confidence_scores):.2%}\n")
    f.write(f"Median Confidence:  {np.median(confidence_scores):.2%}\n")
    f.write(f"Std Deviation:      {np.std(confidence_scores):.2%}\n")
    f.write(f"Min Confidence:     {np.min(confidence_scores):.2%}\n")
    f.write(f"Max Confidence:     {np.max(confidence_scores):.2%}\n\n")
    
    f.write("SAMPLE CLASSIFICATIONS\n")
    f.write("-"*80 + "\n")
    f.write("Top 5 Most Confident Predictions:\n\n")
    top_confident = df.nlargest(5, 'Confidence_Score')
    for idx, row in top_confident.iterrows():
        f.write(f"Comment: {row['clean_text'][:150]}...\n")
        f.write(f"Category: {row['Predicted_Category']}\n")
        f.write(f"Confidence: {row['Confidence_Score']:.2%}\n\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("Top 5 Least Confident Predictions:\n\n")
    low_confident = df.nsmallest(5, 'Confidence_Score')
    for idx, row in low_confident.iterrows():
        f.write(f"Comment: {row['clean_text'][:150]}...\n")
        f.write(f"Category: {row['Predicted_Category']}\n")
        f.write(f"Confidence: {row['Confidence_Score']:.2%}\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("GENERATED FILES\n")
    f.write("="*80 + "\n")
    f.write(f"1. {output_csv}\n")
    f.write(f"2. {output_matrix}\n")
    f.write(f"3. {output_dist}\n")
    f.write(f"4. {output_confidence}\n")
    f.write(f"5. {report_file}\n")
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✅ Saved: {report_file}")

# Summary
print("\n" + "="*80)
print("PHASE 2 COMPLETE! 🎉")
print("="*80)
print("\n📁 Generated Files:")
print(f"   1. {output_csv}")
print(f"   2. {output_matrix}")
print(f"   3. {output_dist}")
print(f"   4. {output_confidence}")
print(f"   5. {report_file}")

print("\n📊 Summary:")
print(f"   ✅ Classified {len(df)} negative comments")
print(f"   ✅ Used BART Zero-Shot Model (facebook/bart-large-mnli)")
print(f"   ✅ Average confidence: {np.mean(confidence_scores):.2%}")
print(f"   ✅ Generated visualizations and detailed report")

print("\n" + "="*80)
print("Phase 2 classification completed successfully!")
print("="*80)

