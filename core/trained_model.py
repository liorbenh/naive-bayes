"""
Simple TrainedModel entity that wraps training results.
"""
from typing import Dict, Any, List

class TrainedModel:
    """
    Simple wrapper for trained model data and metadata.
    """
    
    def __init__(self, model_metrics: Dict[str, Any], features: List[str], classes: List[str]):
        """
        Initialize with model metrics and basic info.
        
        Args:
            model_metrics: Dictionary from ModelTrainer.train()
            features: List of feature names
            classes: List of class names
        """
        # Core model data
        self.log_priors = model_metrics["log_priors"]
        self.log_likelihoods = model_metrics["log_likelihoods"]
        self.vocab_sizes = model_metrics["vocab_sizes"]
        self.vocab_per_feature = model_metrics["vocab_per_feature"]
        self.smoothing = model_metrics["smoothing"]
        
        # Model structure
        self.features = features
        self.classes = classes
        
        # Simple stats
        self.num_features = len(features)
        self.num_classes = len(classes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving/serialization."""
        return {
            "log_priors": self.log_priors,
            "log_likelihoods": self.log_likelihoods,
            "vocab_sizes": self.vocab_sizes,
            "vocab_per_feature": self.vocab_per_feature,
            "smoothing": self.smoothing,
            "features": self.features,
            "classes": self.classes
        }
    
    @classmethod
    def from_dict(cls, model_data: Dict[str, Any]) -> 'TrainedModel':
        """Create TrainedModel from dictionary."""
        model_metrics = {
            "log_priors": model_data["log_priors"],
            "log_likelihoods": model_data["log_likelihoods"],
            "vocab_sizes": model_data["vocab_sizes"],
            "vocab_per_feature": model_data["vocab_per_feature"],
            "smoothing": model_data["smoothing"]
        }
        return cls(model_metrics, model_data["features"], model_data["classes"])