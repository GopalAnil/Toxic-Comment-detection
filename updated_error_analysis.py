import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create output directory - use the same path as train_smaller.py
OUTPUT_DIR = "./output/toxic_detection_improved/error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model and tokenizer - use the same path as train_smaller.py
MODEL_DIR = "./output/toxic_detection_improved/best_model"
print(f"Loading model from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Load test data - larger sample for better analysis
print("Loading test data...")
dataset = load_dataset("civil_comments")

# Create a balanced test set with 100 examples of each class
test_toxic = [ex for ex in dataset["test"] if ex["toxicity"] >= 0.5][:100]
test_non_toxic = [ex for ex in dataset["test"] if ex["toxicity"] < 0.5][:100]
test_examples = test_toxic + test_non_toxic
np.random.shuffle(test_examples)

test_dataset = Dataset.from_list(test_examples)
print(f"Created balanced test set with {len(test_toxic)} toxic and {len(test_non_toxic)} non-toxic examples")

# Process the test data
def process_test_data():
    """Tokenize and predict on test data"""
    all_texts = test_dataset["text"]
    all_labels = [1 if score >= 0.5 else 0 for score in test_dataset["toxicity"]]
    
    all_preds = []
    all_scores = []
    batch_size = 16
    
    # Process in batches to be more efficient
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # For binary classification
            probabilities = torch.softmax(logits, dim=1)
            batch_scores = probabilities[:, 1].cpu().numpy()  # Probability of toxic class
            batch_preds = (batch_scores >= 0.5).astype(int)
            
            all_scores.extend(batch_scores)
            all_preds.extend(batch_preds)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        "text": all_texts,
        "true_toxicity": test_dataset["toxicity"],
        "true_label": all_labels,
        "pred_score": all_scores,
        "pred_label": all_preds
    })
    
    # Save full results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_results.csv"), index=False)
    
    return results_df

# Analyze model behavior across different thresholds
def analyze_thresholds(df):
    """Analyze model performance at different thresholds"""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for threshold in thresholds:
        # Apply threshold
        preds = (df["pred_score"] >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(df["true_label"], preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            df["true_label"], preds, average='binary', zero_division=0
        )
        
        # Create confusion matrix
        cm = confusion_matrix(df["true_label"], preds)
        
        if len(cm) == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where confusion matrix might not be 2x2
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Store results
        results[threshold] = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "confusion_matrix": cm.tolist()
        }
    
    # Find best threshold based on F1 score
    best_threshold = max(results.items(), key=lambda x: x[1]["f1_score"])[0]
    
    # Save threshold results
    with open(os.path.join(OUTPUT_DIR, "threshold_results.json"), "w") as f:
        json.dump({
            "thresholds": results,
            "best_threshold": float(best_threshold),
            "best_f1": float(results[best_threshold]["f1_score"])
        }, f, indent=2)
    
    return results, best_threshold

# Analyze error categories
def analyze_error_categories(df):
    """Analyze errors by category"""
    # Add error type column
    df["error_type"] = "correct"
    df.loc[(df["true_label"] == 0) & (df["pred_label"] == 1), "error_type"] = "false_positive"
    df.loc[(df["true_label"] == 1) & (df["pred_label"] == 0), "error_type"] = "false_negative"
    
    # Get most confident errors
    false_positives = df[df["error_type"] == "false_positive"].sort_values("pred_score", ascending=False).head(10)
    false_negatives = df[df["error_type"] == "false_negative"].sort_values("pred_score", ascending=True).head(10)
    
    # Combine and save
    error_examples = pd.concat([
        false_positives[["text", "true_toxicity", "pred_score", "error_type"]],
        false_negatives[["text", "true_toxicity", "pred_score", "error_type"]]
    ])
    
    error_examples.to_csv(os.path.join(OUTPUT_DIR, "error_examples.csv"), index=False)
    
    # Count errors by length
    df["text_length"] = df["text"].apply(len)
    length_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    df["length_bin"] = pd.cut(df["text_length"], bins=length_bins, labels=["0-50", "51-100", "101-200", "201-500", "501-1000", "1001+"])
    
    # Calculate error rates by length
    error_by_length = df.groupby("length_bin").apply(
        lambda x: pd.Series({
            "total": len(x),
            "false_positives": sum((x["true_label"] == 0) & (x["pred_label"] == 1)),
            "false_negatives": sum((x["true_label"] == 1) & (x["pred_label"] == 0)),
            "fp_rate": sum((x["true_label"] == 0) & (x["pred_label"] == 1)) / max(1, sum(x["true_label"] == 0)),
            "fn_rate": sum((x["true_label"] == 1) & (x["pred_label"] == 0)) / max(1, sum(x["true_label"] == 1))
        })
    ).reset_index()
    
    error_by_length.to_csv(os.path.join(OUTPUT_DIR, "error_by_length.csv"), index=False)
    
    return error_by_length, error_examples

# Generate data for score distribution analysis
def analyze_score_distribution(df):
    """Generate data for score distribution analysis"""
    # Create bins for score distribution
    bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    
    # Create distribution for toxic and non-toxic examples
    toxic_dist = np.histogram(df[df["true_label"] == 1]["pred_score"], bins=bins)[0]
    non_toxic_dist = np.histogram(df[df["true_label"] == 0]["pred_score"], bins=bins)[0]
    
    # Create a dataframe with the distribution data
    dist_df = pd.DataFrame({
        "bin_start": bins[:-1],
        "bin_end": bins[1:],
        "toxic_count": toxic_dist,
        "non_toxic_count": non_toxic_dist
    })
    
    # Save distribution data
    dist_df.to_csv(os.path.join(OUTPUT_DIR, "score_distribution.csv"), index=False)
    
    return dist_df

# Main analysis function
def perform_error_analysis():
    """Perform comprehensive error analysis"""
    print("Starting error analysis...")
    
    # Get predictions on test data
    print("Processing test data and generating predictions...")
    results_df = process_test_data()
    
    # Calculate stats for toxic and non-toxic examples
    toxic_scores = results_df[results_df["true_label"] == 1]["pred_score"]
    non_toxic_scores = results_df[results_df["true_label"] == 0]["pred_score"]
    
    score_stats = {
        "toxic_examples_count": len(toxic_scores),
        "non_toxic_examples_count": len(non_toxic_scores),
        "toxic_mean_score": float(np.mean(toxic_scores)),
        "toxic_median_score": float(np.median(toxic_scores)),
        "toxic_std_score": float(np.std(toxic_scores)),
        "non_toxic_mean_score": float(np.mean(non_toxic_scores)),
        "non_toxic_median_score": float(np.median(non_toxic_scores)),
        "non_toxic_std_score": float(np.std(non_toxic_scores)),
        "score_difference": float(np.mean(toxic_scores) - np.mean(non_toxic_scores)),
        "overall_mean_score": float(np.mean(results_df["pred_score"])),
        "overall_std_score": float(np.std(results_df["pred_score"]))
    }
    
    # Save score statistics
    with open(os.path.join(OUTPUT_DIR, "score_statistics.json"), "w") as f:
        json.dump(score_stats, f, indent=2)
    
    # Analyze thresholds
    print("Analyzing performance across different thresholds...")
    threshold_results, best_threshold = analyze_thresholds(results_df)
    
    # Analyze error categories
    print("Analyzing error categories...")
    error_by_length, error_examples = analyze_error_categories(results_df)
    
    # Generate score distribution data
    print("Generating score distribution data...")
    score_dist = analyze_score_distribution(results_df)
    
    # Get results at best threshold
    best_results = threshold_results[best_threshold]
    
    # Print results summary
    print("\n===== ERROR ANALYSIS RESULTS =====")
    print(f"Analyzed {len(results_df)} test examples")
    print(f"Found {score_stats['toxic_examples_count']} toxic and {score_stats['non_toxic_examples_count']} non-toxic examples")
    
    print("\nScore Distribution:")
    print(f"Toxic comments: Mean={score_stats['toxic_mean_score']:.4f}, Median={score_stats['toxic_median_score']:.4f}, Std={score_stats['toxic_std_score']:.4f}")
    print(f"Non-toxic comments: Mean={score_stats['non_toxic_mean_score']:.4f}, Median={score_stats['non_toxic_median_score']:.4f}, Std={score_stats['non_toxic_std_score']:.4f}")
    print(f"Score difference (toxic - non-toxic): {score_stats['score_difference']:.4f}")
    
    print("\nPerformance with different thresholds:")
    for threshold in sorted(threshold_results.keys()):
        result = threshold_results[threshold]
        print(f"Threshold {threshold}: F1={result['f1_score']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}")
    
    print(f"\nBest threshold: {best_threshold}")
    print(f"With best threshold:")
    print(f"- Accuracy: {best_results['accuracy']:.4f}")
    print(f"- F1 Score: {best_results['f1_score']:.4f}")
    print(f"- Precision: {best_results['precision']:.4f}")
    print(f"- Recall: {best_results['recall']:.4f}")
    print(f"- True Positives: {best_results['true_positives']} / {score_stats['toxic_examples_count']}")
    print(f"- False Positives: {best_results['false_positives']} / {score_stats['non_toxic_examples_count']}")
    print(f"- True Negatives: {best_results['true_negatives']} / {score_stats['non_toxic_examples_count']}")
    print(f"- False Negatives: {best_results['false_negatives']} / {score_stats['toxic_examples_count']}")
    
    print("\nError Analysis by Text Length:")
    print(error_by_length.to_string(index=False))
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return results_df, threshold_results, best_threshold

# Run the analysis
if __name__ == "__main__":
    perform_error_analysis()