"""Pure Naive Bayes classifier for classification only."""
from typing import Dict, Any
import math
from .trained_model import TrainedModel


class Classifier:
    def __init__(self, model: TrainedModel):
        self.model = model

    def classify(self, sample: Dict[str, Any]) -> Any:
        """
        Classify the class using precomputed log probabilities.
        """
        class_scores = {}

        for c in self.model.log_priors:
            log_prob = self.model.log_priors[c]

            for feature, value in sample.items():
                feature_probs = self.model.log_likelihoods[feature][c]

                if value in feature_probs:
                    log_prob += feature_probs[value]
                else:
                    # Handle unseen value (Laplace smoothing)
                    vocab_size = self.model.vocab_sizes[feature]
                    prob = self.model.smoothing / (self.model.smoothing * vocab_size)
                    log_prob += math.log(prob)

            class_scores[c] = log_prob

        return max(class_scores, key=class_scores.get)