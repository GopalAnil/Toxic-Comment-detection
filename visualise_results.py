import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create output directory for visualizations
RESULTS_DIR = "./output/toxic_detection_improved/evaluation"
VISUAL_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(VISUAL_DIR, exist_ok=True)

def load_data():
    """Load evaluation results and prediction data"""
    # Load evaluation results
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
    else:
        eval_results = None
        print(f"Warning: Could not find evaluation results at {eval_path}")
    
    # Load predictions CSV if available
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
    else:
        pred_df = None
        print(f"Warning: Could not find predictions at {pred_path}")
    
    # Load baseline comparison if available
    baseline_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
    else:
        baseline_data = None
        print(f"Warning: Could not find baseline comparison at {baseline_path}")
    
    return eval_results, pred_df, baseline_data

def plot_confusion_matrix(eval_results):
    """Plot confusion matrix"""
    if not eval_results or "confusion_matrix" not in eval_results:
        print("Cannot plot confusion matrix: data not available")
        return
    
    cm = np.array(eval_results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap manually since we're avoiding seaborn
    im = ax.imshow(cm, cmap='Blues')
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-toxic', 'Toxic'])
    ax.set_yticklabels(['Non-toxic', 'Toxic'])
    
    # Add values to cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # Add labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved.")

def plot_score_distribution(pred_df):
    """Plot distribution of prediction scores by class"""
    if pred_df is None:
        print("Cannot plot score distribution: data not available")
        return
    
    # Create separate arrays for toxic and non-toxic scores
    toxic_scores = pred_df[pred_df["true_label"] == 1]["pred_score"]
    non_toxic_scores = pred_df[pred_df["true_label"] == 0]["pred_score"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    bins = np.linspace(0, 1, 20)
    ax.hist(toxic_scores, bins=bins, alpha=0.5, label='Toxic', color='red')
    ax.hist(non_toxic_scores, bins=bins, alpha=0.5, label='Non-toxic', color='blue')
    
    # Add vertical line for default threshold
    ax.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
    ax.axvline(x=0.7, color='green', linestyle='--', label='Optimized Threshold (0.7)')
    
    # Add labels and title
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prediction Scores by Class')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "score_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Score distribution saved.")

def plot_baseline_comparison(baseline_data):
    """Plot comparison between baseline and fine-tuned model"""
    if baseline_data is None:
        print("Cannot plot baseline comparison: data not available")
        return
    
    # Extract metrics for comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    baseline_values = [baseline_data['baseline'][f'eval_{m}'] for m in metrics]
    finetuned_values = [baseline_data['fine_tuned'][f'eval_{m}'] for m in metrics]
    improvements = [baseline_data['improvements'][m] for m in metrics]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    width = 0.35
    x = np.arange(len(metrics))
    
    # Create bars
    baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='lightgray')
    finetuned_bars = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned', color='darkblue')
    
    # Add improvement annotations
    for i, imp in enumerate(improvements):
        ax.annotate(f'+{imp:.4f}' if imp > 0 else f'{imp:.4f}', 
                   xy=(i, max(baseline_values[i], finetuned_values[i]) + 0.02),
                   ha='center', va='bottom',
                   color='green' if imp > 0 else 'red')
    
    # Customize chart
    ax.set_ylabel('Score')
    ax.set_title('Baseline vs. Fine-tuned Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)  # Set y-axis to go from 0 to 1.1 to leave room for annotations
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "baseline_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Baseline comparison saved.")

def plot_threshold_analysis():
    """Plot model performance across different thresholds"""
    # Try to load threshold results from error analysis
    threshold_path = "./output/toxic_detection_improved/error_analysis/threshold_results.json"
    if not os.path.exists(threshold_path):
        print(f"Cannot plot threshold analysis: data not available at {threshold_path}")
        return
    
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
    
    if "thresholds" not in threshold_data:
        print("Cannot plot threshold analysis: invalid data format")
        return
    
    # Extract threshold values and metrics
    thresholds = sorted([float(t) for t in threshold_data["thresholds"].keys()])
    precision = [threshold_data["thresholds"][str(t)]["precision"] for t in thresholds]
    recall = [threshold_data["thresholds"][str(t)]["recall"] for t in thresholds]
    f1 = [threshold_data["thresholds"][str(t)]["f1_score"] for t in thresholds]
    accuracy = [threshold_data["thresholds"][str(t)]["accuracy"] for t in thresholds]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(thresholds, precision, 'o-', label='Precision', color='blue')
    ax.plot(thresholds, recall, 'o-', label='Recall', color='red')
    ax.plot(thresholds, f1, 'o-', label='F1 Score', color='green')
    ax.plot(thresholds, accuracy, 'o-', label='Accuracy', color='purple')
    
    # Add vertical line for best threshold
    best_threshold = float(threshold_data.get("best_threshold", 0.7))
    ax.axvline(x=best_threshold, color='black', linestyle='--', 
               label=f'Best Threshold ({best_threshold})')
    
    # Customize chart
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Across Different Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "threshold_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Threshold analysis saved.")

def main():
    """Generate all visualizations"""
    print("Starting visualization generation...")
    
    # Load data
    eval_results, pred_df, baseline_data = load_data()
    
    # Generate visualizations
    plot_confusion_matrix(eval_results)
    plot_score_distribution(pred_df)
    plot_baseline_comparison(baseline_data)
    plot_threshold_analysis()
    
    print(f"All visualizations saved to {VISUAL_DIR}")

if __name__ == "__main__":
    main()