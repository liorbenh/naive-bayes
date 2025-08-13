"""Configuration management."""
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import httpx

SUPPORTED_DATA_FILE_TYPES = ['.csv']

class Config:
    """Application configuration - reads exclusively from environment variables.
    
    All configuration values must be provided via .env file or environment variables.
    No defaults are provided in code to enforce single source of truth.
    """
    
    def __init__(self):
        # Base directories
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.models_dir = self.data_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Service configuration - directly from env (no defaults in code)
        self.scheme = os.getenv("SCHEME")
        
        self.classifier_host = os.getenv("CLASSIFIER_HOST")
        self.classifier_port = int(os.getenv("CLASSIFIER_PORT"))
        self.classifier_url = httpx.URL(
            scheme=self.scheme, 
            host=self.classifier_host, 
            port=self.classifier_port
        )
        
        self.model_pipeline_manager_host = os.getenv("MODEL_PIPELINE_MANAGER_HOST")
        self.model_pipeline_manager_port = int(os.getenv("MODEL_PIPELINE_MANAGER_PORT"))
        self.model_pipeline_manager_url = httpx.URL(
            scheme=self.scheme, 
            host=self.model_pipeline_manager_host, 
            port=self.model_pipeline_manager_port
        )

        # Model configuration
        self.test_size = float(os.getenv("TEST_SIZE"))
        self.random_state = int(os.getenv("RANDOM_STATE"))
        self.laplace_smoothing = float(os.getenv("LAPLACE_SMOOTHING"))
        
        # Cache configuration
        self.model_cache_capacity = int(os.getenv("MODEL_CACHE_CAPACITY"))
        
        # Data configuration
        self.csv_encoding = os.getenv("CSV_ENCODING")
        
        # Ensure directories exist
        self.ensure_directories_exist()
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to data file."""
        return self.raw_data_dir / filename
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to model file."""
        return self.models_dir / model_name
    
    def get_available_datasets(self) -> Dict[str, str]:
        """Dynamically discover available datasets in the data directory."""
        datasets = {}
        
        if not self.raw_data_dir.exists():
            return datasets
        
        for file_path in self.raw_data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DATA_FILE_TYPES:
                # Use filename without extension as dataset name
                dataset_name = file_path.stem
                datasets[dataset_name] = file_path.name
        
        return datasets
    
    def get_available_dataset_names(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.get_available_datasets().keys())
    
    def ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# Global config instance (singleton)
load_dotenv()
config = Config()