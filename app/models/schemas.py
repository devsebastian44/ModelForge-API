from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from datetime import datetime


class ProblemType(str, Enum):
    """Tipos de problemas de ML"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class Algorithm(str, Enum):
    """Algoritmos disponibles"""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    KMEANS = "kmeans"


class TrainingRequest(BaseModel):
    """Request para entrenar un modelo"""
    dataset: List[Dict[str, Any]] = Field(..., description="Dataset en formato JSON")
    target_column: Optional[str] = Field(None, description="Columna objetivo (no aplica para clustering)")
    problem_type: ProblemType = Field(..., description="Tipo de problema ML")
    algorithm: Algorithm = Field(..., description="Algoritmo a utilizar")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hiperparámetros del modelo")
    
    @validator("dataset")
    def validate_dataset(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Dataset cannot be empty")
        if len(v) > 100000:
            raise ValueError("Dataset too large (max 100,000 rows)")
        return v
    
    @validator("target_column")
    def validate_target_column(cls, v, values):
        if "problem_type" in values:
            if values["problem_type"] != ProblemType.CLUSTERING and not v:
                raise ValueError("target_column is required for supervised learning")
            if values["problem_type"] == ProblemType.CLUSTERING and v:
                raise ValueError("target_column should not be provided for clustering")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset": [
                    {"feature1": 1.2, "feature2": 3.4, "target": 0},
                    {"feature1": 2.1, "feature2": 4.5, "target": 1}
                ],
                "target_column": "target",
                "problem_type": "classification",
                "algorithm": "random_forest",
                "hyperparameters": {"n_estimators": 100, "max_depth": 10}
            }
        }


class MetricsResponse(BaseModel):
    """Métricas de evaluación del modelo"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    silhouette_score: Optional[float] = None
    inertia: Optional[float] = None


class TrainingResponse(BaseModel):
    """Response del entrenamiento"""
    model_id: str = Field(..., description="ID único del modelo entrenado")
    problem_type: ProblemType
    algorithm: Algorithm
    metrics: MetricsResponse
    training_samples: int
    test_samples: int
    feature_names: List[str]
    created_at: datetime
    status: Literal["success", "failed"] = "success"
    message: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request para realizar predicciones"""
    model_id: str = Field(..., description="ID del modelo a utilizar")
    data: List[Dict[str, Any]] = Field(..., description="Datos para predicción")
    
    @validator("data")
    def validate_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Prediction data cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_20240125_123456",
                "data": [
                    {"feature1": 1.5, "feature2": 3.2},
                    {"feature1": 2.3, "feature2": 4.1}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response de predicción"""
    model_id: str
    predictions: List[Any]
    prediction_count: int
    status: Literal["success", "failed"] = "success"


class ModelInfo(BaseModel):
    """Información de un modelo guardado"""
    model_id: str
    problem_type: ProblemType
    algorithm: Algorithm
    feature_names: List[str]
    created_at: datetime
    metrics: MetricsResponse


class ModelListResponse(BaseModel):
    """Lista de modelos disponibles"""
    models: List[ModelInfo]
    total_count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime