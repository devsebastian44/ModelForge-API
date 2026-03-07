from fastapi import APIRouter, Depends, status
from typing import Annotated

from app.models.schemas import TrainingRequest, TrainingResponse
from app.services.ml_service import MLService
from app.api.dependencies import get_ml_service


router = APIRouter(prefix="/training", tags=["Training"])


@router.post(
    "",
    response_model=TrainingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train a new ML model",
    description="""
    Train a new machine learning model with the provided dataset.
    
    Supports:
    - **Regression**: Linear Regression, Random Forest
    - **Classification**: Logistic Regression, Random Forest
    - **Clustering**: K-Means
    
    Returns a unique model_id for future predictions.
    """
)
async def train_model(
    request: TrainingRequest,
    ml_service: Annotated[MLService, Depends(get_ml_service)]
) -> TrainingResponse:
    """
    Entrena un modelo de Machine Learning.
    
    Args:
        request: Configuración del entrenamiento
        ml_service: Servicio de ML inyectado
    
    Returns:
        TrainingResponse con ID del modelo y métricas
    """
    return ml_service.train_model(request)