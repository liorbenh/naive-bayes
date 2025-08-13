"""Tests for data loader module."""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_loader import LocalCSVDataLoader


def test_data_loader_init():
    """Test DataLoader initialization."""
    loader = LocalCSVDataLoader("test.csv")
    assert loader.filename == "test.csv"
    assert "test.csv" in str(loader.file_path)
    
    
def test_data_loader_with_custom_dir():
    """Test DataLoader with custom directory."""
    loader = LocalCSVDataLoader("test.csv", "custom/path")
    assert "custom/path" in str(loader.data_dir)


def test_load_data_file_not_found():
    """Test loading non-existent file raises error."""
    loader = LocalCSVDataLoader("nonexistent_file.csv")
    
    with pytest.raises(FileNotFoundError):
        loader.load_data()