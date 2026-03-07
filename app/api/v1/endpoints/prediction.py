from fastapi import APIRouter, Depends, status
from typing import Annotated

from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.ml_service import MLService
from app.api.dependencies import get_ml_service


router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make predictions with a trained model",
    description="""
    Use a previously trained model to make predictions on new data.
    
    The data must have the same features as the training dataset.
    """
)
async def predict(
    request: PredictionRequest,
    ml_service: Annotated[MLService, Depends(get_ml_service)]
) -> PredictionResponse:
    """
    Realiza predicciones usando un modelo entrenado.
    
    Args:
        request: Datos para predicción y model_id
        ml_service: Servicio de ML inyectado
    
    Returns:
        PredictionResponse con las predicciones
    """
    return ml_service.predict(request)