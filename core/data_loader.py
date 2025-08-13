"""Data loading functionality with interface pattern."""
import pandas as pd
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from config.general import config


class DataLoader(ABC):
    """Abstract interface for data loading."""
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from source."""
        pass

class LocalCSVDataLoader(DataLoader):
    """CSV file data loader implementation."""
    
    def __init__(self, filename: str, data_dir: Optional[str] = None):
        """
        Initialize CSV data loader.
        
        Args:
            filename: Name of the file (e.g., 'data.csv')
            data_dir: Directory containing the file (defaults to config.raw_data_dir)
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        self.data_dir = Path(data_dir) if data_dir else config.raw_data_dir
        self.filename = filename
        self.file_path = self.data_dir / filename
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        try:
            return pd.read_csv(
                self.file_path, 
                encoding=config.csv_encoding,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {e}")