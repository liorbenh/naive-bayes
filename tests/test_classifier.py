"""Tests for classifier module."""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.classifier import Classifier


def test_classifier_init():
    """Test NaiveBayesClassifier initialization."""
    classifier = Classifier()
    assert not classifier.is_trained
    assert classifier.model_data == {}


def test_classifier_classify_before_training():
    """Test that classification before training raises error."""
    classifier = Classifier()
    
    # Create dummy data
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    
    with pytest.raises(ValueError):
        classifier.classify(X)
