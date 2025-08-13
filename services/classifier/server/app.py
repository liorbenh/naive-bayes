"""
Classifier Service - API Controller.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from config.general import config
from core.classifier import Classifier
from core.trained_model import TrainedModel
from utils.lru_cache import LRUCache


# Request/Response Models
class ClassifyRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_id: Optional[str] = None

class ClassifyResponse(BaseModel):
    classifications: List[Any]
    model_id: str


# Service Layer
class ClassifierService:
    """Handles model loading and classification business logic."""
    
    def __init__(self):
        self.model_cache = LRUCache(capacity=config.model_cache_capacity)
    
    async def _fetch_model_data(self, model_id: str) -> Dict[str, Any]:
        """Fetch model data from model pipeline manager service."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.model_pipeline_manager_url}/models/{model_id}")
            if response.status_code != 200:
                raise Exception(f"Model {model_id} not found")
            
            return response.json()["model_data"]
    
    def _create_classifier(self, model_data: Dict[str, Any]) -> Classifier:
        """Create classifier from model data."""
        trained_model = TrainedModel.from_dict(model_data)
        return Classifier(trained_model)
    
    async def get_classifier(self, model_id: str) -> Classifier:
        """Get classifier, loading if not cached."""
        # Check cache first
        cached_classifier = self.model_cache.get(model_id)
        if cached_classifier:
            return cached_classifier
        
        # Load from remote service
        model_data = await self._fetch_model_data(model_id)
        classifier = self._create_classifier(model_data)
        
        # Cache the classifier
        self.model_cache.put(model_id, classifier)
        
        return classifier
    
    def classify_samples(self, classifier: Classifier, samples: List[Dict[str, Any]]) -> List[Any]:
        """Classify multiple samples."""
        return [classifier.classify(sample) for sample in samples]
    
    async def list_available_models(self) -> Dict[str, Any]:
        """Fetch available models from model pipeline manager service."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.model_pipeline_manager_url}/models")
            if response.status_code != 200:
                raise Exception("Model service unavailable")
            return response.json()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self.model_cache.size(),
            "cache_capacity": self.model_cache.capacity,
            "cached_models": self.model_cache.keys()
        }


# Controller Layer
class ClassifierController:
    """Handles HTTP request/response logic and error handling."""
    
    def __init__(self, service: ClassifierService):
        self.service = service
    
    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """Handle classification request."""
        if not request.model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        
        try:
            classifier = await self.service.get_classifier(request.model_id)
            classifications = self.service.classify_samples(classifier, request.data)
            
            return ClassifyResponse(classifications=classifications, model_id=request.model_id)
        
        except Exception as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            else:
                raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Handle health check request."""
        cache_stats = self.service.get_cache_stats()
        return {
            "status": "healthy",
            "service": "classifier",
            **cache_stats
        }
    
    async def list_models(self) -> Dict[str, Any]:
        """Handle list models request."""
        try:
            return await self.service.list_available_models()
        except Exception as e:
            if "unavailable" in str(e):
                raise HTTPException(status_code=503, detail="Model service unavailable")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


# Initialize FastAPI app, service and controller
service = ClassifierService()
controller = ClassifierController(service)
app = FastAPI(title="Classifier Service")


# API Endpoints
@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(request: ClassifyRequest):
    """Classify data samples."""
    return await controller.classify(request)


@app.get("/health")
async def health_endpoint():
    """Service health check."""
    return controller.health_check()


@app.get("/models")
async def list_models_endpoint():
    """List available models."""
    return await controller.list_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.classifier_host, port=config.classifier_port)