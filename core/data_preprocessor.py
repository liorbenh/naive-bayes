"""Data preprocessing functionality."""
import pandas as pd

class DataPreprocessor:
    """Handles data cleaning and preprocessing operations."""
    
    def __init__(self, data:pd.DataFrame) -> None:
        """Initialize the data preprocessing component."""
        self.data = data

    def clean(self):
        # self.data clean actions #
        return self
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess data - main method for data preprocessing."""
        self.clean()
        return self.data