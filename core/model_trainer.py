"""
Model Trainer for Naive Bayes classification with log probabilities.
"""

import math
import pandas as pd
from .trained_model import TrainedModel


class ModelTrainer:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def train(self, df: pd.DataFrame, label_column: str) -> TrainedModel:
        """
        Train Naive Bayes and return a TrainedModel.
        """
        features = [col for col in df.columns if col != label_column]

        # Class priors
        class_counts = df[label_column].value_counts().to_dict()
        total_samples = len(df)
        log_priors = {
            c: math.log(count / total_samples)
            for c, count in class_counts.items()
        }

        # Store log likelihoods
        log_likelihoods = {}
        vocab_per_feature = {}
        vocab_sizes = {}

        for feature in features:
            # Get raw counts without fill_value
            counts_df = df.groupby(label_column)[feature].value_counts()

            vocab = set(df[feature].unique())
            vocab_per_feature[feature] = vocab
            vocab_sizes[feature] = len(vocab)

            # Precompute log probabilities for every (class, value)
            feature_probs = {}
            for c in class_counts:
                # Filter counts for this class
                class_feature_counts = counts_df.get(c, None)
                if isinstance(class_feature_counts, pd.Series):
                    class_feature_counts = class_feature_counts.to_dict()
                else:
                    class_feature_counts = {}

                total_count = sum(class_feature_counts.values())
                denom = total_count + self.smoothing * vocab_sizes[feature]

                feature_probs[c] = {
                    v: math.log((class_feature_counts.get(v, 0) + self.smoothing) / denom)
                    for v in vocab
                }

            log_likelihoods[feature] = feature_probs

        model_metrics = {
            "log_priors": log_priors,
            "log_likelihoods": log_likelihoods,
            "vocab_sizes": vocab_sizes,
            "vocab_per_feature": vocab_per_feature,
            "smoothing": self.smoothing
        }
        
        return TrainedModel(model_metrics, features, list(class_counts.keys()))