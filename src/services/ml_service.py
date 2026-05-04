import numpy as np
from typing import Any, Dict, List
from datetime import datetime

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

from src.models.schemas import (
    TrainingRequest, TrainingResponse, PredictionRequest, 
    PredictionResponse, ProblemType, Algorithm, MetricsResponse
)
from src.services.data_processor import DataProcessor
from src.services.metrics_service import MetricsService
from src.services.model_manager import ModelManager
from src.core.exceptions import MLAPIException, TrainingError, PredictionError
from src.config import get_settings


class MLService:
    """Servicio principal de Machine Learning"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_processor = DataProcessor()
        self.metrics_service = MetricsService()
        self.model_manager = ModelManager()
    
    def _get_model_instance(
        self, 
        algorithm: Algorithm, 
        problem_type: ProblemType,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """Instancia el modelo según el algoritmo"""
        
        # Parámetros base del modelo
        params = {**hyperparameters}
        
        # Agregar random_state solo si es soportado por el algoritmo
        algorithms_with_rs = [
            Algorithm.LOGISTIC_REGRESSION, 
            Algorithm.RANDOM_FOREST, 
            Algorithm.KMEANS
        ]
        
        if algorithm in algorithms_with_rs and "random_state" not in params:
            params["random_state"] = self.settings.RANDOM_STATE
        
        model_map = {
            Algorithm.LINEAR_REGRESSION: LinearRegression,
            Algorithm.LOGISTIC_REGRESSION: LogisticRegression,
            Algorithm.RANDOM_FOREST: (
                RandomForestClassifier if problem_type == ProblemType.CLASSIFICATION 
                else RandomForestRegressor
            ),
            Algorithm.KMEANS: KMeans
        }
        
        # Validar compatibilidad
        if algorithm == Algorithm.LINEAR_REGRESSION and problem_type != ProblemType.REGRESSION:
            raise TrainingError("Linear Regression only works with regression problems")
        
        if algorithm == Algorithm.LOGISTIC_REGRESSION and problem_type != ProblemType.CLASSIFICATION:
            raise TrainingError("Logistic Regression only works with classification problems")
        
        if algorithm == Algorithm.KMEANS and problem_type != ProblemType.CLUSTERING:
            raise TrainingError("K-Means only works with clustering problems")
        
        model_class = model_map.get(algorithm)
        
        if not model_class:
            raise TrainingError(f"Algorithm {algorithm} not supported")
        
        try:
            # Para K-Means, asegurar que haya n_clusters
            if algorithm == Algorithm.KMEANS and "n_clusters" not in params:
                params["n_clusters"] = 3
            
            return model_class(**params)
        except Exception as e:
            raise TrainingError(f"Failed to instantiate model: {str(e)}")
    
    def train_model(self, request: TrainingRequest) -> TrainingResponse:
        """Entrena un modelo de ML"""
        
        try:
            # 1. Procesar datos
            df = self.data_processor.json_to_dataframe(request.dataset)
            self.data_processor.validate_dataset(
                df, request.target_column, request.problem_type
            )
            
            # 2. Preparar features y target
            X, y, feature_names = self.data_processor.prepare_data(
                df, request.target_column, request.problem_type
            )
            
            # 3. Split de datos
            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
            
            # 4. Instanciar y entrenar modelo
            model = self._get_model_instance(
                request.algorithm, 
                request.problem_type,
                request.hyperparameters
            )
            
            if request.problem_type == ProblemType.CLUSTERING:
                model.fit(X_train)
                y_pred = model.predict(X_test)
                metrics = self.metrics_service.calculate_metrics(
                    request.problem_type, None, y_pred, X_test, model
                )
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = self.metrics_service.calculate_metrics(
                    request.problem_type, y_test, y_pred
                )
            
            # 5. Guardar modelo
            model_id = self.model_manager.generate_model_id()
            self.model_manager.save_model(
                model_id=model_id,
                model=model,
                problem_type=request.problem_type,
                algorithm=request.algorithm,
                feature_names=feature_names,
                metrics=metrics,
                label_encoders=self.data_processor.label_encoders
            )
            
            # 6. Retornar respuesta
            return TrainingResponse(
                model_id=model_id,
                problem_type=request.problem_type,
                algorithm=request.algorithm,
                metrics=metrics,
                training_samples=len(X_train),
                test_samples=len(X_test),
                feature_names=feature_names,
                created_at=datetime.now(),
                status="success"
            )
            
        except MLAPIException:
            raise
        except Exception as e:
            raise TrainingError(str(e))
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Realiza predicciones con un modelo entrenado"""
        
        try:
            # 1. Cargar modelo
            model, metadata, encoders = self.model_manager.load_model(request.model_id)
            
            # 2. Preparar datos de predicción
            if encoders:
                self.data_processor.label_encoders = encoders
            
            X_pred = self.data_processor.prepare_prediction_data(
                request.data,
                metadata["feature_names"]
            )
            
            # 3. Realizar predicción
            predictions = model.predict(X_pred)
            
            # 4. Decodificar predicciones si es necesario
            target_col = None
            for key in metadata.get("feature_names", []):
                if key not in request.data[0]:
                    target_col = key
                    break
            
            if encoders and target_col and target_col in encoders:
                predictions = encoders[target_col].inverse_transform(predictions.astype(int))
            
            return PredictionResponse(
                model_id=request.model_id,
                predictions=predictions.tolist(),
                prediction_count=len(predictions),
                status="success"
            )
            
        except MLAPIException:
            raise
        except Exception as e:
            raise PredictionError(str(e))
