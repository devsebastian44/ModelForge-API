from fastapi import HTTPException, status


class MLAPIException(HTTPException):
    """Excepción base para la API"""
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)


class DatasetValidationError(MLAPIException):
    """Error en validación de dataset"""
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Dataset validation failed: {detail}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class ModelNotFoundError(MLAPIException):
    """Modelo no encontrado"""
    def __init__(self, model_id: str):
        super().__init__(
            detail=f"Model '{model_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND
        )


class TrainingError(MLAPIException):
    """Error durante el entrenamiento"""
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Training failed: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class PredictionError(MLAPIException):
    """Error durante la predicción"""
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Prediction failed: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )