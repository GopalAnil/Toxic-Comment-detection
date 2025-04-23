import os
import json
import numpy as np
import pandas as pd
import random
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_curve, 
    auc, 
    confusion_matrix
)

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load the best model
MODEL_DIR = "./output/toxic_detection_improved/best_model"
RESULTS_DIR = "./output/toxic_detection_improved/evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check if binary classification model
model_info_path = os.path.join(MODEL_DIR, "model_info.json")
if os.path.exists(model_info_path):
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
    is_binary = model_info.get("model_type") == "binary_classification"
    num_labels = model_info.get("num_labels", 2)
else:
    # Default to binary if no info
    is_binary = True
    num_labels = 2

print(f"Model type: {'Binary Classification' if is_binary else 'Regression'}")

# Load a balanced test set
print("Loading test dataset...")
dataset = load_dataset("civil_comments")

# Create a balanced test set
test_toxic = [ex for ex in dataset["test"] if ex["toxicity"] >= 0.5][:100]
test_non_toxic = [ex for ex in dataset["test"] if ex["toxicity"] < 0.5][:100]
test_examples = test_toxic + test_non_toxic
random.shuffle(test_examples)

test_dataset = Dataset.from_list(test_examples)
print(f"Created balanced test set with {len(test_toxic)} toxic and {len(test_non_toxic)} non-toxic examples")

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=num_labels)

# Preprocess the test data
def preprocess(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

print("Preprocessing test data...")
encoded_test = test_dataset.map(preprocess, batched=True)

# Convert toxicity scores to binary labels
def binarize_labels(examples):
    return {"labels": [1 if score >= 0.5 else 0 for score in examples["toxicity"]]}

encoded_test = encoded_test.map(binarize_labels, batched=True)
encoded_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define metrics function
def compute_metrics(pred):
    if is_binary:
        # For binary classification
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
    else:
        # For regression
        labels = pred.label_ids
        preds = pred.predictions.squeeze()
        # Convert to binary
        preds = (preds >= 0.5).astype(int)
        labels = (labels >= 0.5).astype(int)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # Try to compute AUC
    try:
        if is_binary:
            # For binary classification, use softmax to get probabilities
            logits = torch.tensor(pred.predictions)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # Class 1 probability
            fpr, tpr, _ = roc_curve(labels, probs)
        else:
            fpr, tpr, _ = roc_curve(labels, pred.predictions.squeeze())
        auc_score = auc(fpr, tpr)
    except Exception as e:
        print(f"Error computing AUC: {e}")
        auc_score = 0.0
        
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }

# ====================== EVALUATION OF FINE-TUNED MODEL ======================

# Initialize trainer just for evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

trainer = Trainer(
    model=model,
    eval_dataset=encoded_test,
    compute_metrics=compute_metrics
)

# Evaluate the model
print("Evaluating model on test data...")
results = trainer.evaluate()

# Get predictions for detailed analysis
print("Getting predictions for detailed analysis...")
predictions = trainer.predict(encoded_test)

# Extract predictions and labels
if is_binary:
    # For binary classification
    logits = torch.tensor(predictions.predictions)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # Class 1 probability
    preds = predictions.predictions.argmax(-1)
else:
    # For regression
    probs = predictions.predictions.squeeze()
    preds = (probs >= 0.5).astype(int)

labels = predictions.label_ids

# Calculate confusion matrix
cm = confusion_matrix(labels, preds)

# ====================== BASELINE MODEL COMPARISON ======================

print("Loading baseline model for comparison...")
# Load the original pre-trained model (without fine-tuning)
baseline_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",  # Original pre-trained model
    num_labels=num_labels
)
baseline_model.to(device)

# Initialize trainer for baseline evaluation
baseline_trainer = Trainer(
    model=baseline_model,
    eval_dataset=encoded_test,
    compute_metrics=compute_metrics
)

# Evaluate the baseline model
print("Evaluating baseline model on test data...")
baseline_results = baseline_trainer.evaluate()

# Calculate improvements from baseline to fine-tuned
improvements = {
    "accuracy": results["eval_accuracy"] - baseline_results["eval_accuracy"],
    "precision": results["eval_precision"] - baseline_results["eval_precision"],
    "recall": results["eval_recall"] - baseline_results["eval_recall"],
    "f1": results["eval_f1"] - baseline_results["eval_f1"],
    "auc": results.get("eval_auc", 0.0) - baseline_results.get("eval_auc", 0.0)
}

# ====================== SAVE RESULTS ======================

# Save detailed results
evaluation_results = {
    "fine_tuned_metrics": results,
    "baseline_metrics": baseline_results,
    "improvements": improvements,
    "confusion_matrix": cm.tolist(),
    "is_binary_classification": is_binary
}

with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
    json.dump(evaluation_results, f, indent=2)

# Save prediction data to CSV for later visualization if needed
pred_df = pd.DataFrame({
    'text': test_dataset["text"],
    'true_toxicity': test_dataset["toxicity"],
    'true_label': labels,
    'pred_score': probs,
    'pred_label': preds
})
pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

# Save text-based results
tn, fp, fn, tp = cm.ravel()
with open(os.path.join(RESULTS_DIR, "confusion_matrix.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(f"True Negative: {tn}, False Positive: {fp}\n")
    f.write(f"False Negative: {fn}, True Positive: {tp}\n")

# Save baseline comparison
with open(os.path.join(RESULTS_DIR, "baseline_comparison.json"), "w") as f:
    json.dump({
        "baseline": baseline_results,
        "fine_tuned": results,
        "improvements": improvements
    }, f, indent=2)

# ====================== PRINT RESULTS ======================

# Print results summary
print("\n===== EVALUATION RESULTS =====")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall: {results['eval_recall']:.4f}")
print(f"F1 Score: {results['eval_f1']:.4f}")
print(f"AUC: {results['eval_auc']:.4f}")
print(f"\nConfusion Matrix:")
print(f"[[{cm[0,0]}, {cm[0,1]}],")
print(f" [{cm[1,0]}, {cm[1,1]}]]")

# Print baseline comparison
print("\n===== BASELINE VS FINE-TUNED COMPARISON =====")
print(f"Metric      | Baseline    | Fine-tuned  | Improvement")
print(f"------------|-------------|-------------|------------")
print(f"Accuracy    | {baseline_results['eval_accuracy']:.4f} | {results['eval_accuracy']:.4f} | {improvements['accuracy']:.4f}")
print(f"Precision   | {baseline_results['eval_precision']:.4f} | {results['eval_precision']:.4f} | {improvements['precision']:.4f}")
print(f"Recall      | {baseline_results['eval_recall']:.4f} | {results['eval_recall']:.4f} | {improvements['recall']:.4f}")
print(f"F1 Score    | {baseline_results['eval_f1']:.4f} | {results['eval_f1']:.4f} | {improvements['f1']:.4f}")
print(f"AUC         | {baseline_results.get('eval_auc', 0.0):.4f} | {results.get('eval_auc', 0.0):.4f} | {improvements['auc']:.4f}")

print(f"\nResults saved to {RESULTS_DIR}")