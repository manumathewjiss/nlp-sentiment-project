import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import os


class Config:
    OUTPUT_DIR = "outputs"
    BERTWEET_IMBALANCED_RESULTS = "phase1_sentiment_results_imbalanced.csv"
    BERTWEET_BALANCED_RESULTS = "phase1_sentiment_results_balanced.csv"
    ROBERTA_BALANCED_RESULTS = "phase1_sentiment_results_roberta.csv"
    ROBERTA_IMBALANCED_RESULTS = "phase1_sentiment_results_roberta_imbalanced.csv"
    COMPARISON_REPORT = "model_comparison_report.txt"
    COMPARISON_ACCURACY_CHART = "model_comparison_accuracy.png"
    COMPARISON_DETAILED = "model_comparison_detailed.csv"
    COMPARISON_CONFUSION = "model_comparison_confusion_matrices.png"


def load_results():
    print("\n" + "="*70)
    print("LOADING MODEL RESULTS")
    print("="*70)
    
    results = {}
    
    imbalanced_path = os.path.join(Config.OUTPUT_DIR, Config.BERTWEET_IMBALANCED_RESULTS)
    if os.path.exists(imbalanced_path):
        results['bertweet_imbalanced'] = pd.read_csv(imbalanced_path)
        print(f"✓ Loaded BERTweet (Imbalanced): {len(results['bertweet_imbalanced']):,} comments")
    else:
        print(f"✗ Missing: {imbalanced_path}")
        results['bertweet_imbalanced'] = None
    
    balanced_path = os.path.join(Config.OUTPUT_DIR, Config.BERTWEET_BALANCED_RESULTS)
    if os.path.exists(balanced_path):
        results['bertweet_balanced'] = pd.read_csv(balanced_path)
        print(f"✓ Loaded BERTweet (Balanced): {len(results['bertweet_balanced']):,} comments")
    else:
        print(f"✗ Missing: {balanced_path}")
        results['bertweet_balanced'] = None
    
    roberta_balanced_path = os.path.join(Config.OUTPUT_DIR, Config.ROBERTA_BALANCED_RESULTS)
    if os.path.exists(roberta_balanced_path):
        results['roberta_balanced'] = pd.read_csv(roberta_balanced_path)
        print(f"✓ Loaded RoBERTa (Balanced): {len(results['roberta_balanced']):,} comments")
    else:
        print(f"✗ Missing: {roberta_balanced_path}")
        print(f"   Run 'python roberta_main.py' first to generate RoBERTa results")
        results['roberta_balanced'] = None
    
    roberta_imbalanced_path = os.path.join(Config.OUTPUT_DIR, Config.ROBERTA_IMBALANCED_RESULTS)
    if os.path.exists(roberta_imbalanced_path):
        results['roberta_imbalanced'] = pd.read_csv(roberta_imbalanced_path)
        print(f"✓ Loaded RoBERTa (Imbalanced): {len(results['roberta_imbalanced']):,} comments")
    else:
        print(f"✗ Missing: {roberta_imbalanced_path}")
        print(f"   Run 'python roberta_imbalanced_main.py' first to generate RoBERTa imbalanced results")
        results['roberta_imbalanced'] = None
    
    return results


def calculate_metrics(df):
    if df is None:
        return None
    
    y_true = df['Sentiment']
    y_pred = df['Predicted_Sentiment']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=['negative', 'neutral', 'positive'],
        average=None
    )
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    metrics = {
        'accuracy': accuracy,
        'precision': {
            'negative': precision[0],
            'neutral': precision[1],
            'positive': precision[2],
            'macro': macro_precision
        },
        'recall': {
            'negative': recall[0],
            'neutral': recall[1],
            'positive': recall[2],
            'macro': macro_recall
        },
        'f1': {
            'negative': f1[0],
            'neutral': f1[1],
            'positive': f1[2],
            'macro': macro_f1
        },
        'support': {
            'negative': support[0],
            'neutral': support[1],
            'positive': support[2],
            'total': len(df)
        }
    }
    
    return metrics


def compare_models(results):
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    metrics = {}
    
    if results['bertweet_imbalanced'] is not None:
        metrics['BERTweet (Imbalanced)'] = calculate_metrics(results['bertweet_imbalanced'])
        print("✓ Calculated metrics for BERTweet (Imbalanced)")
    
    if results['bertweet_balanced'] is not None:
        metrics['BERTweet (Balanced)'] = calculate_metrics(results['bertweet_balanced'])
        print("✓ Calculated metrics for BERTweet (Balanced)")
    
    if results['roberta_balanced'] is not None:
        metrics['RoBERTa (Balanced)'] = calculate_metrics(results['roberta_balanced'])
        print("✓ Calculated metrics for RoBERTa (Balanced)")
    
    if results['roberta_imbalanced'] is not None:
        metrics['RoBERTa (Imbalanced)'] = calculate_metrics(results['roberta_imbalanced'])
        print("✓ Calculated metrics for RoBERTa (Imbalanced)")
    
    return metrics


def generate_report(metrics):
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("        MODEL COMPARISON REPORT")
    report_lines.append("  BERTweet vs RoBERTa | Imbalanced vs Balanced Datasets")
    report_lines.append("="*70)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_lines.append("\n" + "─"*70)
    report_lines.append("1. OVERALL ACCURACY COMPARISON")
    report_lines.append("─"*70)
    
    accuracies = []
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            acc = model_metrics['accuracy']
            total = model_metrics['support']['total']
            accuracies.append((model_name, acc, total))
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  Dataset Size: {total:,} comments")
            report_lines.append(f"  Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    
    if accuracies:
        best_model = max(accuracies, key=lambda x: x[1])
        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"WINNER (Highest Accuracy): {best_model[0]}")
        report_lines.append(f"Accuracy: {best_model[1]*100:.2f}%")
        report_lines.append(f"{'='*70}")
    
    report_lines.append("\n\n" + "─"*70)
    report_lines.append("2. PER-CLASS PERFORMANCE COMPARISON")
    report_lines.append("─"*70)
    
    for sentiment_class in ['negative', 'neutral', 'positive']:
        report_lines.append(f"\n{sentiment_class.upper()} Sentiment:")
        report_lines.append(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        report_lines.append("-"*70)
        
        for model_name, model_metrics in metrics.items():
            if model_metrics:
                prec = model_metrics['precision'][sentiment_class]
                rec = model_metrics['recall'][sentiment_class]
                f1 = model_metrics['f1'][sentiment_class]
                report_lines.append(f"{model_name:<30} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
    
    report_lines.append("\n\n" + "─"*70)
    report_lines.append("3. MACRO-AVERAGED METRICS")
    report_lines.append("─"*70)
    report_lines.append(f"\n{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    report_lines.append("-"*70)
    
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            prec = model_metrics['precision']['macro']
            rec = model_metrics['recall']['macro']
            f1 = model_metrics['f1']['macro']
            report_lines.append(f"{model_name:<30} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
    
    report_lines.append("\n\n" + "─"*70)
    report_lines.append("4. KEY INSIGHTS")
    report_lines.append("─"*70)
    
    if 'BERTweet (Imbalanced)' in metrics and 'BERTweet (Balanced)' in metrics:
        imb_acc = metrics['BERTweet (Imbalanced)']['accuracy']
        bal_acc = metrics['BERTweet (Balanced)']['accuracy']
        diff = imb_acc - bal_acc
        
        report_lines.append("\n📊 Dataset Balance Effect (BERTweet model):")
        report_lines.append(f"   Imbalanced Dataset: {imb_acc*100:.2f}%")
        report_lines.append(f"   Balanced Dataset:   {bal_acc*100:.2f}%")
        report_lines.append(f"   Difference:         {diff*100:+.2f}% ({abs(diff)*100:.2f} percentage points)")
        
        if diff > 0:
            report_lines.append(f"   → Imbalanced dataset achieved HIGHER accuracy")
            report_lines.append(f"      (likely due to majority class bias)")
        else:
            report_lines.append(f"   → Balanced dataset achieved HIGHER accuracy")
            report_lines.append(f"      (better generalization across all classes)")
    
    if 'BERTweet (Balanced)' in metrics and 'RoBERTa (Balanced)' in metrics:
        bert_acc = metrics['BERTweet (Balanced)']['accuracy']
        rob_acc = metrics['RoBERTa (Balanced)']['accuracy']
        diff = rob_acc - bert_acc
        
        report_lines.append("\n\n🤖 Model Architecture Comparison (Balanced Dataset):")
        report_lines.append(f"   BERTweet:  {bert_acc*100:.2f}%")
        report_lines.append(f"   RoBERTa:   {rob_acc*100:.2f}%")
        report_lines.append(f"   Difference: {diff*100:+.2f}% ({abs(diff)*100:.2f} percentage points)")
        
        if diff > 0:
            report_lines.append(f"   → RoBERTa performs BETTER than BERTweet")
            report_lines.append(f"      RoBERTa correctly predicted {int(diff * metrics['RoBERTa (Balanced)']['support']['total'])} more comments")
        else:
            report_lines.append(f"   → BERTweet performs BETTER than RoBERTa")
    
    report_lines.append("\n\n📈 Class-Specific Performance:")
    
    for sentiment_class in ['negative', 'neutral', 'positive']:
        f1_scores = []
        for model_name, model_metrics in metrics.items():
            if model_metrics:
                f1_scores.append((model_name, model_metrics['f1'][sentiment_class]))
        
        if f1_scores:
            best = max(f1_scores, key=lambda x: x[1])
            report_lines.append(f"\n   {sentiment_class.capitalize()} Detection:")
            report_lines.append(f"      Best Model: {best[0]} (F1: {best[1]:.3f})")
    
    report_lines.append("\n\n" + "─"*70)
    report_lines.append("5. RECOMMENDATIONS")
    report_lines.append("─"*70)
    
    if accuracies:
        best_overall = max(accuracies, key=lambda x: x[1])
        report_lines.append(f"\n✅ Recommended Model: {best_overall[0]}")
        report_lines.append(f"   Accuracy: {best_overall[1]*100:.2f}%")
        report_lines.append(f"   Reason: Highest overall accuracy")
        
        if 'RoBERTa (Balanced)' in metrics:
            rob_f1 = metrics['RoBERTa (Balanced)']['f1']['macro']
            report_lines.append(f"\n   RoBERTa also provides:")
            report_lines.append(f"   • Balanced performance across all sentiment classes")
            report_lines.append(f"   • Better handling of neutral sentiments")
            report_lines.append(f"   • More recent training data (2018-2021)")
    
    report_lines.append("\n\n" + "="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    report_path = os.path.join(Config.OUTPUT_DIR, Config.COMPARISON_REPORT)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Report saved: {report_path}")
    print("\n" + "\n".join(report_lines))
    
    return report_lines


def create_accuracy_chart(metrics):
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    models = []
    accuracies = []
    colors = []
    
    color_map = {
        'BERTweet (Imbalanced)': '#FF6B6B',
        'BERTweet (Balanced)': '#4ECDC4',
        'RoBERTa (Balanced)': '#45B7D1',
        'RoBERTa (Imbalanced)': '#95E1D3'
    }
    
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            models.append(model_name)
            accuracies.append(model_metrics['accuracy'] * 100)
            colors.append(color_map.get(model_name, '#95E1D3'))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Sentiment Analysis Accuracy\nYouTube Comments Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    chart_path = os.path.join(Config.OUTPUT_DIR, Config.COMPARISON_ACCURACY_CHART)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy chart saved: {chart_path}")
    plt.close()


def create_detailed_comparison(results, metrics):
    detailed_data = []
    
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            detailed_data.append({
                'Model': model_name,
                'Class': 'Overall',
                'Accuracy': f"{model_metrics['accuracy']:.4f}",
                'Precision': f"{model_metrics['precision']['macro']:.4f}",
                'Recall': f"{model_metrics['recall']['macro']:.4f}",
                'F1-Score': f"{model_metrics['f1']['macro']:.4f}",
                'Support': model_metrics['support']['total']
            })
            
            for sentiment_class in ['negative', 'neutral', 'positive']:
                detailed_data.append({
                    'Model': model_name,
                    'Class': sentiment_class.capitalize(),
                    'Accuracy': '-',
                    'Precision': f"{model_metrics['precision'][sentiment_class]:.4f}",
                    'Recall': f"{model_metrics['recall'][sentiment_class]:.4f}",
                    'F1-Score': f"{model_metrics['f1'][sentiment_class]:.4f}",
                    'Support': model_metrics['support'][sentiment_class]
                })
    
    df_detailed = pd.DataFrame(detailed_data)
    
    detailed_path = os.path.join(Config.OUTPUT_DIR, Config.COMPARISON_DETAILED)
    df_detailed.to_csv(detailed_path, index=False)
    print(f"✓ Detailed comparison saved: {detailed_path}")
    
    return df_detailed


def create_confusion_comparison(results):
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    titles = ['BERTweet (Imbalanced)', 'BERTweet (Balanced)', 
              'RoBERTa (Balanced)', 'RoBERTa (Imbalanced)']
    datasets = ['bertweet_imbalanced', 'bertweet_balanced', 
                'roberta_balanced', 'roberta_imbalanced']
    
    for idx, (ax, title, dataset_key) in enumerate(zip(axes, titles, datasets)):
        if results[dataset_key] is not None:
            df = results[dataset_key]
            y_true = df['Sentiment']
            y_pred = df['Predicted_Sentiment']
            
            cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Neg', 'Neu', 'Pos'],
                       yticklabels=['Neg', 'Neu', 'Pos'],
                       cbar=True)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual' if idx % 2 == 0 else '')
            ax.set_xlabel('Predicted')
        else:
            ax.text(0.5, 0.5, 'Data Not Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    confusion_path = os.path.join(Config.OUTPUT_DIR, Config.COMPARISON_CONFUSION)
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {confusion_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("MODEL COMPARISON ANALYSIS")
    print("BERTweet vs RoBERTa | Imbalanced vs Balanced Datasets")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        results = load_results()
        
        missing_models = []
        if results['roberta_balanced'] is None:
            missing_models.append("RoBERTa (Balanced)")
        if results['roberta_imbalanced'] is None:
            missing_models.append("RoBERTa (Imbalanced)")
        
        if missing_models:
            print("\n" + "!"*70)
            print(f"WARNING: {', '.join(missing_models)} results not found!")
            print("Comparison will proceed with available data only.")
            print("!"*70)
        
        metrics = compare_models(results)
        
        if not metrics:
            print("\nERROR: No model results found!")
            print("Please ensure at least one model has been run.")
            return
        
        generate_report(metrics)
        create_accuracy_chart(metrics)
        create_confusion_comparison(results)
        create_detailed_comparison(results, metrics)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nOutput files created in '{Config.OUTPUT_DIR}/':")
        print(f"  - {Config.COMPARISON_REPORT}")
        print(f"  - {Config.COMPARISON_ACCURACY_CHART}")
        print(f"  - {Config.COMPARISON_CONFUSION}")
        print(f"  - {Config.COMPARISON_DETAILED}")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
