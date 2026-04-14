import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)
from typing import Any, Optional

from src.models.schemas import ProblemType, MetricsResponse


class MetricsService:
    """Servicio para calcular métricas de evaluación"""
    
    def calculate_metrics(
        self,
        problem_type: ProblemType,
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        model: Any = None
    ) -> MetricsResponse:
        """Calcula métricas según el tipo de problema"""
        
        if problem_type == ProblemType.CLASSIFICATION:
            return self._classification_metrics(y_true, y_pred)
        elif problem_type == ProblemType.REGRESSION:
            return self._regression_metrics(y_true, y_pred)
        elif problem_type == ProblemType.CLUSTERING:
            return self._clustering_metrics(X, y_pred, model)
        else:
            return MetricsResponse()
    
    def _classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> MetricsResponse:
        """Métricas para clasificación"""
        
        # Determinar si es binario o multiclase
        n_classes = len(np.unique(y_true))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        return MetricsResponse(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, average=average, zero_division=0))
        )
    
    def _regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> MetricsResponse:
        """Métricas para regresión"""
        
        mse = mean_squared_error(y_true, y_pred)
        
        return MetricsResponse(
            rmse=float(np.sqrt(mse)),
            r2_score=float(r2_score(y_true, y_pred))
        )
    
    def _clustering_metrics(
        self, 
        X: np.ndarray, 
        labels: np.ndarray,
        model: Any
    ) -> MetricsResponse:
        """Métricas para clustering"""
        
        metrics = MetricsResponse()
        
        # Silhouette score (requiere al menos 2 clusters)
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and n_clusters < len(X):
            try:
                metrics.silhouette_score = float(silhouette_score(X, labels))
            except:
                pass
        
        # Inertia (solo para K-Means)
        if hasattr(model, 'inertia_'):
            metrics.inertia = float(model.inertia_)
        
        return metrics
