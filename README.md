# Toxic Comment Detection

A machine learning project that fine-tunes a RoBERTa model to detect toxic comments in online text.

## Overview

This project demonstrates the process of fine-tuning a pre-trained language model (RoBERTa) for toxic comment classification. The model is trained on the Civil Comments dataset to distinguish between toxic and non-toxic content with high accuracy and precision.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Error Analysis](#error-analysis)
- [Future Improvements](#future-improvements)

## Installation

### Requirements

To run this project, you'll need Python 3.8+ and the following packages:

```
transformers>=4.20.0
torch>=1.10.0
datasets>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

### Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/toxic-comment-detection.git
cd toxic-comment-detection
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Dataset

This project uses the Civil Comments dataset, which contains user-generated comments from online platforms with human-annotated toxicity scores. The dataset is accessed via the Hugging Face Datasets library.

For training efficiency, we use a balanced subset:
- 500 examples for training (250 toxic, 250 non-toxic)
- 50 examples for validation (25 toxic, 25 non-toxic)
- 200 examples for testing (100 toxic, 100 non-toxic)

## Project Structure

- `train_smaller.py`: Handles model training and hyperparameter optimization
- `evaluate.py`: Evaluates the model and compares with baseline
- `inference.py`: Provides a simple interface for making predictions
- `updated_error_analysis.py`: Analyzes model errors and optimizes decision threshold
- `visualize_results.py`: Generates visualizations of model performance and analysis
- `requirements.txt`: Lists all required dependencies
- `output/`: Contains trained models and evaluation results

## Usage

### Training

To train the model with hyperparameter optimization:

```
python train_smaller.py
```

This script will:
1. Load and preprocess the Civil Comments dataset
2. Create balanced subsets for training, validation, and testing
3. Train RoBERTa with 3 different learning rates
4. Save the best model to the output directory

### Evaluation

To evaluate the fine-tuned model and compare with baseline:

```
python evaluate.py
```

This script will:
1. Load the best fine-tuned model
2. Evaluate it on a balanced test set
3. Compare with the pre-trained baseline model
4. Save detailed metrics and analysis results

### Inference

To use the model for making predictions:

```
python inference.py
```

This provides an example of using the `ToxicityDetector` class for real-time toxicity detection.

### Error Analysis

To analyze model performance and optimize the decision threshold:

```
python updated_error_analysis.py
```

This script conducts a detailed analysis of model errors and identifies an optimal decision threshold.

## Model Performance

The fine-tuned model shows significant improvements over the baseline:

| Metric      | Baseline | Fine-tuned | Improvement |
|-------------|----------|------------|-------------|
| Accuracy    | 50.00%   | 74.50%     | +24.50%     |
| Precision   | 50.00%   | 67.88%     | +17.88%     |
| F1 Score    | 66.67%   | 78.48%     | +11.81%     |
| AUC         | 51.92%   | 87.31%     | +35.39%     |

With threshold optimization (0.7), we achieve:
- F1 Score: 0.8142
- Precision: 0.7302
- Recall: 0.9200

## Error Analysis

Key findings from error analysis:
- Shorter texts (0-50 characters) have higher false positive rates (50%)
- Text length correlates inversely with error rate
- An optimized threshold of 0.7 (vs. default 0.5) provides better precision with minimal impact on recall

##  Visualization
To generate visualizations of the results:
python visualize_results.py
This script creates four key visualizations:

Confusion Matrix - Shows classification performance
Score Distribution - Displays prediction score distributions by class
Baseline Comparison - Compares baseline and fine-tuned model performance
Threshold Analysis - Shows performance metrics across different thresholds

All visualizations are saved to ./output/toxic_detection_improved/evaluation/visualizations/

## Future Improvements

Potential areas for improvement:
1. Training on a larger dataset to capture more linguistic patterns
2. Implementing data augmentation techniques
3. Extending to multi-class classification for different types of toxicity
4. Improving performance on shorter texts
5. Exploring model ensembles for higher accuracy
6. Adding explainability features to highlight influential text segments

## Why It Matters

This toxicity detection system has applications in both social media platforms and workplace environments. For social media, it helps create safer online spaces and protect vulnerable users from harassment. In professional settings, it can safeguard communication channels and maintain productive collaboration environments. The focus on reducing false positives while maintaining high detection rates addresses a critical challenge in content moderation systems where over-filtering can restrict legitimate speech while under-filtering exposes users to harm.

