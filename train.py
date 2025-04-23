# train_smaller.py
import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Create output directories
base_output_dir = "./output/toxic_detection_improved"
os.makedirs(base_output_dir, exist_ok=True)
os.makedirs(f"{base_output_dir}/results", exist_ok=True)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("civil_comments")

# Create a balanced subset with fewer examples
print("Creating balanced dataset (250 + 250 examples)...")
# Get toxic examples with a faster approach
train_data = dataset["train"]
toxic_indices = [i for i, ex in enumerate(train_data) if ex["toxicity"] >= 0.5][:250]
non_toxic_indices = [i for i, ex in enumerate(train_data) if ex["toxicity"] < 0.5][:250]

# Get the actual examples
toxic_examples = [train_data[i] for i in toxic_indices]
non_toxic_examples = [train_data[i] for i in non_toxic_indices]

# Combine and shuffle
balanced_examples = toxic_examples + non_toxic_examples
random.shuffle(balanced_examples)

print(f"Created balanced dataset with {len(toxic_examples)} toxic and {len(non_toxic_examples)} non-toxic examples")

# Create smaller validation and test sets
valid_toxic = [ex for ex in dataset["validation"] if ex["toxicity"] >= 0.5][:25]
valid_non_toxic = [ex for ex in dataset["validation"] if ex["toxicity"] < 0.5][:25]
test_toxic = [ex for ex in dataset["test"] if ex["toxicity"] >= 0.5][:25]
test_non_toxic = [ex for ex in dataset["test"] if ex["toxicity"] < 0.5][:25]

# Combine and shuffle validation and test sets
valid_examples = valid_toxic + valid_non_toxic
test_examples = test_toxic + test_non_toxic
random.shuffle(valid_examples)
random.shuffle(test_examples)

print(f"Validation set: {len(valid_toxic)} toxic, {len(valid_non_toxic)} non-toxic")
print(f"Test set: {len(test_toxic)} toxic, {len(test_non_toxic)} non-toxic")

# Convert to datasets format
from datasets import Dataset
train_dataset = Dataset.from_list(balanced_examples)
valid_dataset = Dataset.from_list(valid_examples)
test_dataset = Dataset.from_list(test_examples)

# Tokenization
print("Loading tokenizer...")
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# Process all splits and convert to binary classification
print("Preprocessing dataset and binarizing labels...")
encoded = {}

for split_name, split_dataset in [("train", train_dataset), ("validation", valid_dataset), ("test", test_dataset)]:
    # First tokenize
    tokenized = split_dataset.map(preprocess, batched=True)
    
    # Then convert toxicity scores to binary labels (0 or 1)
    def binarize_labels(examples):
        # Convert toxicity scores to binary labels
        return {"labels": [1 if score >= 0.5 else 0 for score in examples["toxicity"]]}
    
    # Apply binarization
    tokenized = tokenized.map(binarize_labels, batched=True)
    
    # Format for PyTorch
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Store in encoded dictionary
    encoded[split_name] = tokenized

# Verify label distribution
for split in encoded:
    labels = encoded[split]["labels"].numpy()
    num_toxic = (labels == 1).sum()
    num_non_toxic = (labels == 0).sum()
    print(f"{split} - Toxic: {num_toxic}, Non-toxic: {num_non_toxic}")

# Define metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Use argmax for classification
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define hyperparameter configurations
hp_configs = [
    # Configuration 1: Baseline
    {
        "name": "baseline",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "weight_decay": 0.01
    },
    # Configuration 2: Higher learning rate
    {
        "name": "high_lr",
        "learning_rate": 5e-5,
        "batch_size": 16, 
        "epochs": 3,
        "weight_decay": 0.01
    },
    # Configuration 3: Lower learning rate
    {
        "name": "low_lr",
        "learning_rate": 1e-5,
        "batch_size": 16,
        "epochs": 3,
        "weight_decay": 0.01
    }
]

# Store results for comparison
all_results = {}

# Run training for each configuration
for hp in hp_configs:
    print(f"\n{'='*50}")
    print(f"Training model with configuration: {hp['name']}")
    print(f"{'='*50}")
    
    # Create model output directory
    model_output_dir = f"{base_output_dir}/{hp['name']}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize model for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        logging_steps=20,  # More frequent logging with smaller dataset
        save_steps=50,
        logging_dir=f"{model_output_dir}/logs"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print(f"Starting training for {hp['name']}...")
    train_result = trainer.train()
    
    # Save the model
    trainer.save_model(f"{model_output_dir}/final")
    
    # Evaluate on test set
    print(f"Evaluating {hp['name']} on test set...")
    test_results = trainer.evaluate(encoded["test"])
    
    # Save results
    results = {
        "hyperparameters": hp,
        "training_metrics": {k: v for k, v in train_result.metrics.items()},
        "test_metrics": test_results
    }
    
    all_results[hp["name"]] = results
    
    # Save individual config results
    with open(f"{base_output_dir}/results/{hp['name']}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results for {hp['name']}:")
    print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"  Test F1 Score: {test_results['eval_f1']:.4f}")
    print(f"  Test Precision: {test_results['eval_precision']:.4f}")
    print(f"  Test Recall: {test_results['eval_recall']:.4f}")

# Compare configurations
print("\n===== HYPERPARAMETER COMPARISON =====")
comparison = {
    "configs": [],
    "train_loss": [],
    "test_accuracy": [],
    "test_f1": [],
    "test_precision": [],
    "test_recall": []
}

for name, results in all_results.items():
    comparison["configs"].append(name)
    comparison["train_loss"].append(results["training_metrics"]["train_loss"])
    comparison["test_accuracy"].append(results["test_metrics"]["eval_accuracy"])
    comparison["test_f1"].append(results["test_metrics"]["eval_f1"])
    comparison["test_precision"].append(results["test_metrics"]["eval_precision"])
    comparison["test_recall"].append(results["test_metrics"]["eval_recall"])

# Find best configuration based on F1 score
best_config_idx = np.argmax(comparison["test_f1"])
best_config = comparison["configs"][best_config_idx]

print(f"Best configuration: {best_config}")
print(f"Best Test Accuracy: {comparison['test_accuracy'][best_config_idx]:.4f}")
print(f"Best Test F1: {comparison['test_f1'][best_config_idx]:.4f}")
print(f"Best Test Precision: {comparison['test_precision'][best_config_idx]:.4f}")
print(f"Best Test Recall: {comparison['test_recall'][best_config_idx]:.4f}")

# Save comparison results to CSV
comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv(f"{base_output_dir}/results/hyperparameter_comparison.csv", index=False)

# Copy the best model to a "best_model" directory
best_model_dir = f"{base_output_dir}/best_model"
os.makedirs(best_model_dir, exist_ok=True)
import shutil
for file in os.listdir(f"{base_output_dir}/{best_config}/final"):
    shutil.copy(f"{base_output_dir}/{best_config}/final/{file}", f"{best_model_dir}/{file}")

# Also save the tokenizer with the best model
tokenizer.save_pretrained(best_model_dir)

# Save model configuration info for inference
with open(f"{best_model_dir}/model_info.json", "w") as f:
    json.dump({
        "model_type": "binary_classification",
        "num_labels": 2,
        "label_mapping": {
            "0": "non-toxic",
            "1": "toxic"
        }
    }, f, indent=2)

print(f"Training and hyperparameter search completed. Results saved to {base_output_dir}")
print(f"Best model saved to {best_model_dir}")