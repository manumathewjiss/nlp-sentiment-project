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


class Config:
    INPUT_CSV = "YoutubeCommentsDataSet.csv"
    OUTPUT_DIR = "outputs"
    FULL_RESULTS = "phase1_sentiment_results_roberta_imbalanced.csv"
    NEGATIVE_COMMENTS = "phase2_input_negative_comments_roberta_imbalanced.csv"
    CONFUSION_MATRIX_IMG = "phase1_confusion_matrix_roberta_imbalanced.png"
    ACCURACY_REPORT = "phase1_accuracy_report_roberta_imbalanced.txt"
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    CACHE_DIR = "./model_cache"
    BATCH_SIZE = 16
    MAX_LENGTH = 128


def load_dataset(file_path):
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Successfully loaded {len(df):,} comments")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nFirst 3 comments preview:")
    for i in range(min(3, len(df))):
        comment = df.loc[i, 'Comment'][:80]
        sentiment = df.loc[i, 'Sentiment']
        print(f"  {i+1}. [{sentiment}] {comment}...")
    
    print(f"\nOriginal sentiment distribution:")
    sentiment_counts = df['Sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize():10s}: {count:5,} ({percentage:5.2f}%)")
    
    return df


def preprocess_data(df):
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING COMMENTS")
    print("="*70)
    
    print(f"Cleaning {len(df):,} comments...")
    print("   (Removing URLs, mentions, hashtags, emojis, special characters)")
    
    tqdm.pandas(desc="Cleaning")
    df['clean_text'] = df['Comment'].progress_apply(clean_text)
    
    initial_count = len(df)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"Removed {removed} empty comments after cleaning")
    
    print(f"Preprocessing complete! {len(df):,} valid comments ready")
    
    print(f"\nBefore/After examples:")
    for i in range(min(2, len(df))):
        original = df.loc[i, 'Comment'][:60]
        cleaned = df.loc[i, 'clean_text'][:60]
        print(f"\n  Example {i+1}:")
        print(f"    Original: {original}...")
        print(f"    Cleaned:  {cleaned}...")
    
    return df


def load_model():
    print("\n" + "="*70)
    print("STEP 3: LOADING ROBERTA MODEL")
    print("="*70)
    
    print(f"Model: {Config.MODEL_NAME}")
    print("   (Using cached model from previous run)")
    print(f"   Cache directory: {Config.CACHE_DIR}")
    
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    print("Tokenizer loaded")
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, cache_dir=Config.CACHE_DIR)
    print("Model loaded")
    
    labels = ['negative', 'neutral', 'positive']
    
    print(f"\nModel Info:")
    print(f"   Architecture: RoBERTa (Robustly Optimized BERT)")
    print(f"   Training: 124M tweets (2018-2021)")
    print(f"   Input: Text (max {Config.MAX_LENGTH} tokens)")
    print(f"   Output: One of {labels}")
    print(f"   Batch size: {Config.BATCH_SIZE} comments at once")
    
    return tokenizer, model, labels


def analyze_sentiment(df, tokenizer, model, labels):
    print("\n" + "="*70)
    print("STEP 4: ANALYZING SENTIMENT WITH ROBERTA")
    print("="*70)
    
    print(f"Processing {len(df):,} comments in batches of {Config.BATCH_SIZE}")
    print("   This will take 10-12 minutes...")
    
    texts = df['clean_text'].tolist()
    results = []
    
    model.eval()
    
    total_batches = (len(texts) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    print(f"   Total batches: {total_batches:,}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), Config.BATCH_SIZE), 
                     desc="Analyzing", 
                     unit="batch"):
            
            batch = texts[i:i+Config.BATCH_SIZE]
            
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
            
            outputs = model(**tokens)
            probs = F.softmax(outputs.logits, dim=1)
            predictions = [labels[p.argmax().item()] for p in probs]
            
            results.extend(predictions)
    
    df['Predicted_Sentiment'] = results
    
    print(f"\nSentiment analysis complete!")
    print(f"   Processed {len(df):,} comments")
    
    print(f"\nPredicted sentiment distribution:")
    pred_counts = df['Predicted_Sentiment'].value_counts()
    for sentiment, count in pred_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize():10s}: {count:5,} ({percentage:5.2f}%)")
    
    return df


def evaluate_model(df):
    print("\n" + "="*70)
    print("STEP 5: EVALUATING MODEL ACCURACY")
    print("="*70)
    
    y_true = df['Sentiment']
    y_pred = df['Predicted_Sentiment']
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"   (RoBERTa correctly predicted {accuracy:.2%} of comments)")
    
    print(f"\nDetailed Classification Report:")
    print("-" * 70)
    report = classification_report(y_true, y_pred, 
                                   target_names=['negative', 'neutral', 'positive'],
                                   digits=3)
    print(report)
    
    print("\nMetric Explanations:")
    print("   Precision: Of predicted positives, how many were correct?")
    print("   Recall: Of actual positives, how many did we find?")
    print("   F1-Score: Balance between precision and recall")
    print("   Support: Number of actual instances")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    print(cm)
    print("\n   Rows = Actual, Columns = Predicted")
    print("   Diagonal = Correct predictions")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Sentiment Analysis Confusion Matrix\nRoBERTa-Twitter on YouTube Comments (Imbalanced)')
    plt.ylabel('Actual Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.tight_layout()
    
    output_path = os.path.join(Config.OUTPUT_DIR, Config.CONFUSION_MATRIX_IMG)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved: {output_path}")
    plt.close()
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def save_results(df, metrics):
    print("\n" + "="*70)
    print("STEP 6: SAVING RESULTS")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {Config.OUTPUT_DIR}/")
    
    full_path = os.path.join(Config.OUTPUT_DIR, Config.FULL_RESULTS)
    df.to_csv(full_path, index=False)
    print(f"Full results saved: {full_path}")
    print(f"   ({len(df):,} comments with predictions)")
    
    negative_df = df[df['Predicted_Sentiment'] == 'negative'].copy()
    negative_path = os.path.join(Config.OUTPUT_DIR, Config.NEGATIVE_COMMENTS)
    negative_df.to_csv(negative_path, index=False)
    print(f"Negative comments saved: {negative_path}")
    print(f"   ({len(negative_df):,} negative comments for Phase 2)")
    
    report_path = os.path.join(Config.OUTPUT_DIR, Config.ACCURACY_REPORT)
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PHASE 1: SENTIMENT ANALYSIS - ROBERTA (IMBALANCED)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {Config.INPUT_CSV}\n")
        f.write(f"Total Comments: {len(df):,}\n")
        f.write(f"Model: {Config.MODEL_NAME}\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-"*70 + "\n")
        f.write(metrics['classification_report'])
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))
    
    print(f"Accuracy report saved: {report_path}")
    print(f"\nAll output files are in: {Config.OUTPUT_DIR}/")


def display_summary(df, metrics, start_time):
    print("\n" + "="*70)
    print("ROBERTA (IMBALANCED) PHASE 1 COMPLETE!")
    print("="*70)
    
    runtime = (datetime.now() - start_time).total_seconds()
    minutes = int(runtime // 60)
    seconds = int(runtime % 60)
    
    print(f"\nTotal Runtime: {minutes}m {seconds}s")
    print(f"Comments Analyzed: {len(df):,}")
    print(f"Model Accuracy: {metrics['accuracy']:.2%}")
    
    print(f"\nSentiment Distribution:")
    sentiment_counts = df['Predicted_Sentiment'].value_counts()
    for sentiment in ['positive', 'neutral', 'negative']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize():10s}: {count:5,} ({percentage:5.2f}%)")
    
    negative_count = sentiment_counts.get('negative', 0)
    print(f"\nNext Step: Update model comparison")
    print(f"   {negative_count:,} negative comments ready for toxicity analysis")
    
    print(f"\nOutput Files:")
    print(f"   - {Config.OUTPUT_DIR}/{Config.FULL_RESULTS}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.NEGATIVE_COMMENTS}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.CONFUSION_MATRIX_IMG}")
    print(f"   - {Config.OUTPUT_DIR}/{Config.ACCURACY_REPORT}")
    
    print("\n" + "="*70)
    print("RoBERTa (Imbalanced) sentiment analysis completed successfully!")
    print("="*70 + "\n")


def main():
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("YOUTUBE COMMENT SENTIMENT ANALYSIS - PHASE 1")
    print("MODEL: ROBERTA-TWITTER (IMBALANCED DATASET)")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: RoBERTa-Twitter (cardiffnlp)")
    print(f"Dataset: {Config.INPUT_CSV} (IMBALANCED)")
    print("="*70)
    
    try:
        df = load_dataset(Config.INPUT_CSV)
        df = preprocess_data(df)
        tokenizer, model, labels = load_model()
        df = analyze_sentiment(df, tokenizer, model, labels)
        metrics = evaluate_model(df)
        save_results(df, metrics)
        display_summary(df, metrics, start_time)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTip: Make sure YoutubeCommentsDataSet.csv is in the same folder!")
        raise


if __name__ == "__main__":
    main()

