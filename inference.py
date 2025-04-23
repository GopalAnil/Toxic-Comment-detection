# inference.py
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToxicityDetector:
    def __init__(self, model_path="./output/toxic_detection_improved/best_model"):
        """Initialize the toxicity detector with a fine-tuned model"""
        # Load model info
        model_info_path = os.path.join(model_path, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                self.model_info = json.load(f)
            self.is_binary = self.model_info.get("model_type") == "binary_classification"
            self.num_labels = self.model_info.get("num_labels", 2)
        else:
            # Default to binary classification if no info file
            self.is_binary = True
            self.num_labels = 2
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=self.num_labels
        )
        self.model.eval()
        
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"Toxicity detector initialized using {self.device}")
        print(f"Model type: {'Binary Classification' if self.is_binary else 'Regression'}")
    
    def predict(self, text):
        """
        Predict toxicity of a single text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary with toxicity score and prediction
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            if self.is_binary:
                # For binary classification
                probabilities = torch.softmax(logits, dim=1)
                toxic_prob = probabilities[0, 1].item()  # Probability of toxic class
                prediction = 1 if toxic_prob >= 0.5 else 0
            else:
                # For regression
                score = torch.sigmoid(logits).item()
                toxic_prob = score
                prediction = 1 if score >= 0.5 else 0
        
        # Return results
        return {
            "text": text,
            "toxicity_score": toxic_prob,
            "is_toxic": prediction == 1,
            "confidence": toxic_prob if prediction == 1 else 1 - toxic_prob
        }
    
    def predict_batch(self, texts):
        """
        Predict toxicity for a batch of texts
        
        Args:
            texts (list): List of strings to analyze
            
        Returns:
            list: List of dictionaries with toxicity scores and predictions
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            if self.is_binary:
                # For binary classification
                probabilities = torch.softmax(logits, dim=1)
                toxic_probs = probabilities[:, 1].cpu().numpy()  # Probabilities of toxic class
                predictions = (toxic_probs >= 0.5).astype(int)
            else:
                # For regression
                scores = torch.sigmoid(logits).cpu().numpy().flatten()
                toxic_probs = scores
                predictions = (scores >= 0.5).astype(int)
        
        # Return results
        results = []
        for i, (text, prob, pred) in enumerate(zip(texts, toxic_probs, predictions)):
            results.append({
                "text": text,
                "toxicity_score": float(prob),
                "is_toxic": bool(pred == 1),
                "confidence": float(prob if pred == 1 else 1 - prob)
            })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = ToxicityDetector()
    
    # Test with some examples
    test_texts = [
        "I love this product, it works great!",
        "You are an idiot and should be fired.",
        "The service was okay, but could be better.",
        "I'm going to find where you live and hurt you.",
        "Thanks for your help with my question."
    ]
    
    # Predict toxicity for each example
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Toxicity score: {result['toxicity_score']:.4f}")
        print(f"Is toxic: {result['is_toxic']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 50)
    
    # Example of batch prediction
    print("\nBatch prediction:")
    results = detector.predict_batch(test_texts)
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['toxicity_score']:.4f}, Is toxic: {result['is_toxic']}")