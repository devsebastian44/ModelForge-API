from fastapi import APIRouter, Depends, status
from typing import Annotated

from src.models.schemas import ModelListResponse, ModelInfo
from src.services.model_manager import ModelManager
from src.api.dependencies import get_model_manager


router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all trained models",
    description="Get a list of all available trained models with their metadata."
)
async def list_models(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
) -> ModelListResponse:
    """
    Lista todos los modelos entrenados.
    
    Args:
        model_manager: Gestor de modelos inyectado
    
    Returns:
        Lista de modelos con su información
    """
    models = model_manager.list_models()
    return ModelListResponse(
        models=models,
        total_count=len(models)
    )


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Get detailed information about a specific model."
)
async def get_model_info(
    model_id: str,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
) -> ModelInfo:
    """
    Obtiene información de un modelo específico.
    
    Args:
        model_id: ID del modelo
        model_manager: Gestor de modelos inyectado
    
    Returns:
        Información del modelo
    """
    _, metadata, _ = model_manager.load_model(model_id)
    
    from datetime import datetime
    from src.models.schemas import ProblemType, Algorithm, MetricsResponse
    
    return ModelInfo(
        model_id=metadata["model_id"],
        problem_type=ProblemType(metadata["problem_type"]),
        algorithm=Algorithm(metadata["algorithm"]),
        feature_names=metadata["feature_names"],
        created_at=datetime.fromisoformat(metadata["created_at"]),
        metrics=MetricsResponse(**metadata["metrics"])
    )


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a model",
    description="Delete a trained model from storage."
)
async def delete_model(
    model_id: str,
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
) -> None:
    """
    Elimina un modelo entrenado.
    
    Args:
        model_id: ID del modelo a eliminar
        model_manager: Gestor de modelos inyectado
    """
    model_manager.delete_model(model_id)
