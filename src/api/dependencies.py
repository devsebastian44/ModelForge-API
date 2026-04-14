from functools import lru_cache
from src.services.ml_service import MLService
from src.services.model_manager import ModelManager


@lru_cache()
def get_ml_service() -> MLService:
    """Dependency injection para MLService"""
    return MLService()


@lru_cache()
def get_model_manager() -> ModelManager:
    """Dependency injection para ModelManager"""
    return ModelManager()
