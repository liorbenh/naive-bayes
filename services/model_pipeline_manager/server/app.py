"""
Model Pipeline Manager Service - Training and model building.
"""

import sys
import os
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Literal, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from sklearn.model_selection import train_test_split
import pandas as pd

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from config.general import config, SUPPORTED_DATA_FILE_TYPES
from core.data_loader import LocalCSVDataLoader
from core.data_preprocessor import DataPreprocessor
from core.model_trainer import ModelTrainer
from core.model_evaluator import ModelEvaluator
from core.classifier import Classifier
from core.trained_model import TrainedModel
from utils.lru_cache import LRUCache


# Request/Response Models with Discriminated Union
class InlineDataSource(BaseModel):
    """Create model from provided inline data."""
    source_type: Literal["inline"] = "inline"
    data: List[Dict[str, Any]]

class DatabaseSource(BaseModel):
    """Create model from database query."""
    source_type: Literal["database"] = "database"
    database_url: str
    query: Optional[str] = None

class LocalFileSource(BaseModel):
    """Create model from local file upload."""
    source_type: Literal["file"] = "file"
    file_path: str
    file_type: Literal["csv"] = "csv"
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path and extension."""
        path = Path(v)
        
        if not path.exists():
            raise ValueError(f"File not found: {v}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        
        if path.suffix.lower() not in SUPPORTED_DATA_FILE_TYPES:
            raise ValueError(f"Unsupported file type. Supported: {SUPPORTED_DATA_FILE_TYPES}. Got: {path.suffix}")
        
        return v

class CreateModelRequest(BaseModel):
    """Request to create a model with discriminated union for data source."""
    target_column: str
    model_name: Optional[str] = None
    source: Union[InlineDataSource, DatabaseSource, LocalFileSource] = Field(discriminator="source_type")
    
    # Flag for training strategy
    train_on_full_data: bool = Field(
        default=True, 
        description="If True, train final model on full dataset after evaluation. If False, only use train split."
    )

class CreateModelResponse(BaseModel):
    """Response model for model creation."""
    model_id: str
    status: str
    message: str
    metrics: Dict[str, Any]
    
    # Additional info about training approach
    training_info: Dict[str, Any] = Field(default_factory=dict)

class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    name: str
    created_at: str
    metrics: Dict[str, Any]
    features: List[str]
    classes: List[str]
    source_type: str
    target_column: str
    feature_count: int
    total_samples: Optional[int] = None

class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[ModelInfo]
    total_count: int

class DatasetInfo(BaseModel):
    """Dataset information response."""
    name: str
    filename: str
    file_type: str
    supported: bool

class DatasetListResponse(BaseModel):
    """Response model for listing datasets."""
    datasets: List[DatasetInfo]
    total_count: int

class GetModelResponse(BaseModel):
    """Response model for getting a specific model."""
    model_id: str
    model_data: Dict[str, Any]
    metadata: Dict[str, Any]
    cached: bool
    # Quick access fields from metadata
    target_column: Optional[str] = None
    feature_count: Optional[int] = None
    total_samples: Optional[int] = None
    model_name: Optional[str] = None


# Service Layer
class ModelPipelineManagerService:
    """Handles model management business logic."""
    
    def __init__(self):
        self.model_cache = LRUCache(capacity=config.model_cache_capacity)
        self.models_dir = config.models_dir
        self.ensure_models_directory()
    
    def ensure_models_directory(self):
        """Ensure models directory exists."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID."""
        return f"model_{uuid.uuid4().hex[:8]}"
    
    def _save_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Save model metadata to JSON file."""
        metadata_path = self.models_dir / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _save_model_data(self, model_id: str, trained_model: TrainedModel) -> None:
        """Save trained model data to pickle file."""
        model_path = self.models_dir / f"{model_id}.pkl"
        model_data = trained_model.to_dict()
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model_data(self, model_id: str) -> Dict[str, Any]:
        """Load model data from pickle file."""
        model_path = self.models_dir / f"{model_id}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Load model metadata from JSON file."""
        metadata_path = self.models_dir / f"{model_id}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _get_available_model_ids(self) -> List[str]:
        """Get list of available model IDs."""
        model_ids = []
        for file in self.models_dir.glob("*_metadata.json"):
            model_id = file.stem.replace("_metadata", "")
            model_ids.append(model_id)
        return model_ids

    
    def _train_model_pipeline(self, df: pd.DataFrame, target_column: str, train_on_full_data: bool = True) -> Tuple[TrainedModel, Dict[str, Any],  Dict[str, Any]]:
        """Execute the complete training pipeline with optional full-data training."""
        # Preprocess data
        preprocessor = DataPreprocessor(df)
        df_processed = preprocessor.preprocess()
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Do train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y
        )
        
        # Step 1: Train evaluation model on training data only
        train_df = X_train.copy()
        train_df[target_column] = y_train
        
        trainer = ModelTrainer(smoothing=config.laplace_smoothing)
        evaluation_model = trainer.train(train_df, target_column)
        
        # Step 2: Evaluate on test data
        test_df = X_test.copy()
        test_df[target_column] = y_test
        
        classifier = Classifier(evaluation_model)
        evaluator = ModelEvaluator(classifier)
        eval_metrics = evaluator.evaluate(test_df, target_column)
        
        # Add additional information to metrics
        eval_metrics.update({
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "total_samples": len(df_processed),
            "test_size_used": config.test_size,
            "random_state_used": config.random_state,
            "target_column": target_column,
            "feature_count": len(X.columns),
            "feature_names": list(X.columns)
        })
        
        # Step 3: Decide which model to return
        if train_on_full_data:
            # Train final model on ALL data
            print("Training final model on full dataset...")
            final_trainer = ModelTrainer(smoothing=config.laplace_smoothing)
            final_model = final_trainer.train(df_processed, target_column)
            
            training_info = {
                "evaluation_model_samples": len(train_df),
                "final_model_samples": len(df_processed),
                "evaluation_strategy": "train_test_split",
                "final_training": "full_dataset",
                "performance_note": "Metrics are from evaluation model, final model may perform better",
                "target_column": target_column,
                "feature_count": len(X.columns),
                "features_used": list(X.columns)
            }
            
            return final_model, eval_metrics, training_info
        else:
            # Return the evaluation model
            training_info = {
                "model_samples": len(train_df),
                "evaluation_strategy": "train_test_split", 
                "final_training": "train_split_only",
                "performance_note": "Model and metrics are from same training data",
                "target_column": target_column,
                "feature_count": len(X.columns),
                "features_used": list(X.columns)
            }
            
            return evaluation_model, eval_metrics, training_info

    def _load_data(self, source: Union[InlineDataSource, DatabaseSource, LocalFileSource]) -> Tuple[pd.DataFrame, str]:
        """Load data from any source type."""
        
        if isinstance(source, InlineDataSource):
            return pd.DataFrame(source.data), "inline_data"
            
        elif isinstance(source, LocalFileSource):
            filename = Path(source.file_path).name
            data_dir = str(Path(source.file_path).parent)
            data_loader = LocalCSVDataLoader(filename, data_dir)
            return data_loader.load_data(), source.file_path
            
        elif isinstance(source, DatabaseSource):
            raise NotImplementedError("Database data loader not yet implemented")
            
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    async def create_model(self, request: CreateModelRequest) -> CreateModelResponse:
        """Create a new model from different sources."""
        try:
            model_id = self._generate_model_id()
            model_name = request.model_name or f"Model_{model_id}"
            
            # Load data
            df, data_source = self._load_data(request.source)
            
            # Validate target column
            if request.target_column not in df.columns:
                available_columns = list(df.columns)
                raise ValueError(
                    f"Target column '{request.target_column}' not found in data. "
                    f"Available columns: {available_columns}"
                )
            
            # Train model with optional full-data training
            trained_model, evaluation_metrics, training_info = self._train_model_pipeline(
                df, 
                request.target_column, 
                train_on_full_data=request.train_on_full_data
            )
            
            # Save model and metadata
            self._save_model_data(model_id, trained_model)
            
            metadata = {
                "model_id": model_id,
                "name": model_name,
                "created_at": datetime.now().isoformat(),
                "data_source": data_source,
                "source_type": request.source.source_type,
                "target_column": request.target_column,
                "features": trained_model.features,
                "classes": trained_model.classes,
                "metrics": evaluation_metrics,
                "training_info": training_info,
                "total_samples": len(df),
                "smoothing": config.laplace_smoothing,
                "train_on_full_data": request.train_on_full_data
            }
            self._save_model_metadata(model_id, metadata)
            
            return CreateModelResponse(
                model_id=model_id,
                status="success",
                message=f"Model '{model_name}' created successfully",
                metrics=evaluation_metrics,
                training_info=training_info
            )
            
        except Exception as e:
            return CreateModelResponse(
                model_id="",
                status="error", 
                message=f"Model creation failed: {str(e)}",
                metrics={},
                training_info={}
            )
    
    def get_model(self, model_id: str) -> GetModelResponse:
        """Get model data with LRU caching."""
        # Check cache first
        cached_model = self.model_cache.get(model_id)
        if cached_model:
            metadata = cached_model["metadata"]
            return GetModelResponse(
                model_id=model_id,
                model_data=cached_model["model_data"],
                metadata=metadata,
                cached=True,
                target_column=metadata.get("target_column"),
                feature_count=len(metadata.get("features", [])),
                total_samples=metadata.get("total_samples"),
                model_name=metadata.get("name")
            )
        
        # Load from disk
        try:
            model_data = self._load_model_data(model_id)
            metadata = self._load_model_metadata(model_id)
            
            # Create response object
            response = GetModelResponse(
                model_id=model_id,
                model_data=model_data,
                metadata=metadata,
                cached=False,
                target_column=metadata.get("target_column"),
                feature_count=len(metadata.get("features", [])),
                total_samples=metadata.get("total_samples"),
                model_name=metadata.get("name")
            )
            
            # Cache the result (store as dict for caching)
            cache_data = {
                "model_data": model_data,
                "metadata": metadata
            }
            self.model_cache.put(model_id, cache_data)
            
            return response
            
        except FileNotFoundError:
            raise Exception(f"Model {model_id} not found")
    
    def list_models(self) -> ModelListResponse:
        """List all available models."""
        models = []
        model_ids = self._get_available_model_ids()
        
        for model_id in model_ids:
            try:
                metadata = self._load_model_metadata(model_id)
                model_info = ModelInfo(
                    model_id=metadata["model_id"],
                    name=metadata["name"],
                    created_at=metadata["created_at"],
                    metrics=metadata["metrics"],
                    features=metadata["features"],
                    classes=metadata["classes"],
                    source_type=metadata.get("source_type", "unknown"),
                    target_column=metadata.get("target_column", "unknown"),
                    feature_count=len(metadata.get("features", [])),
                    total_samples=metadata.get("total_samples")
                )
                models.append(model_info)
            except Exception:
                # Skip corrupted models
                continue
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return ModelListResponse(
            models=models,
            total_count=len(models)
        )
    
    def list_datasets(self) -> DatasetListResponse:
        """List available datasets in the data directory."""
        datasets = []
        
        try:
            available_datasets = config.get_available_datasets()
            
            for dataset_name, filename in available_datasets.items():
                file_path = config.get_data_path(filename)
                file_extension = file_path.suffix.lower()
                
                # Use the same constant
                supported = file_extension in SUPPORTED_DATA_FILE_TYPES
                
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    filename=filename,
                    file_type=file_extension.lstrip('.') if file_extension else 'unknown',
                    supported=supported
                )
                datasets.append(dataset_info)
        
        except Exception as e:
            # Log the error but return empty list rather than failing
            print(f"Error listing datasets: {e}")
        
        return DatasetListResponse(
            datasets=datasets,
            total_count=len(datasets)
        )

    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self.model_cache.size(),
            "cache_capacity": self.model_cache.capacity,
            "cached_models": self.model_cache.keys()
        }


# Controller Layer
class ModelPipelineManagerController:
    """Handles HTTP request/response logic and error handling."""
    
    def __init__(self, service: ModelPipelineManagerService):
        self.service = service
    
    async def create_model(self, request: CreateModelRequest) -> CreateModelResponse:
        """Handle model creation request."""
        try:
            response = await self.service.create_model(request)
            if response.status == "error":
                raise HTTPException(status_code=400, detail=response.message)
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model creation failed: {str(e)}")
    
    def get_model(self, model_id: str) -> GetModelResponse:
        """Handle get model request."""
        try:
            return self.service.get_model(model_id)
        except Exception as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            else:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")
    
    def list_models(self) -> ModelListResponse:
        """Handle list models request."""
        try:
            return self.service.list_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
    
    def list_datasets(self) -> DatasetListResponse:
        """Handle list datasets request."""
        try:
            return self.service.list_datasets()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Handle health check request."""
        try:
            cache_stats = self.service.get_cache_stats()
            
            return {
                "status": "healthy",
                "service": "model_pipeline_manager",
                **cache_stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "model_pipeline_manager",
                "error": str(e)
            }


# Initialize FastAPI app, service and controller
service = ModelPipelineManagerService()
controller = ModelPipelineManagerController(service)
app = FastAPI(title="Model Pipeline Manager Service")


# API Endpoints
@app.post("/models/create", response_model=CreateModelResponse)
async def create_model_endpoint(request: CreateModelRequest):
    """Create a model from various data sources (inline data, database, or file)."""
    return await controller.create_model(request)


@app.get("/models/{model_id}", response_model=GetModelResponse)
async def get_model_endpoint(model_id: str):
    """Get model data by ID."""
    return controller.get_model(model_id)


@app.get("/models", response_model=ModelListResponse)
async def list_models_endpoint():
    """List all available models."""
    return controller.list_models()


@app.get("/datasets", response_model=DatasetListResponse)
async def list_datasets_endpoint():
    """List available datasets in the data directory with file type support info."""
    return controller.list_datasets()


@app.get("/health")
async def health_endpoint():
    """Service health check."""
    return controller.health_check()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.model_pipeline_manager_host, port=config.model_pipeline_manager_port)