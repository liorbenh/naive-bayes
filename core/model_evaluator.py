"""Model evaluation functionality."""
import pandas as pd
from typing import Dict, Any

class ModelEvaluator:
    def __init__(self, classifier):
        """
        :param classifier: An instance of NaiveBayesClassifier
        """
        self.classifier = classifier

    def evaluate(self, data: pd.DataFrame, label_column: str) -> Dict[str, Any]:
        """
        Evaluate model on a dataset using vectorized pandas operations.
        :param data: DataFrame with features + label column.
        :param label_column: Name of the label column.
        :return: Dict with essential evaluation metrics.
        """
        # Get features and true labels
        features = [col for col in data.columns if col != label_column]
        y_true = data[label_column]
        
        # Vectorized prediction using apply
        feature_data = data[features]
        y_pred = feature_data.apply(
            lambda row: self.classifier.classify(row.to_dict()), 
            axis=1
        )
        
        # Calculate basic metrics
        accuracy = (y_true == y_pred).mean()
        total_samples = len(y_true)
        correct_predictions = (y_true == y_pred).sum()
        
        return {
            "accuracy": float(accuracy),
            "correct": int(correct_predictions),
            "total": int(total_samples)
        }