import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.models.schemas import ProblemType
from app.core.exceptions import DatasetValidationError
from app.config import get_settings


class DataProcessor:
    """Servicio de procesamiento de datos"""
    
    def __init__(self):
        self.settings = get_settings()
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def json_to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convierte JSON a DataFrame"""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise DatasetValidationError("Dataset is empty")
            return df
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse dataset: {str(e)}")
    
    def validate_dataset(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str], 
        problem_type: ProblemType
    ) -> None:
        """Valida el dataset"""
        
        # Validar que no esté vacío
        if df.empty:
            raise DatasetValidationError("Dataset is empty")
        
        # Validar columna objetivo para aprendizaje supervisado
        if problem_type != ProblemType.CLUSTERING:
            if not target_column:
                raise DatasetValidationError("target_column is required for supervised learning")
            if target_column not in df.columns:
                raise DatasetValidationError(f"Target column '{target_column}' not found in dataset")
        
        # Validar valores nulos
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            raise DatasetValidationError(f"Dataset contains null values in columns: {null_cols}")
        
        # Validar que haya suficientes datos
        if len(df) < 10:
            raise DatasetValidationError("Dataset too small (minimum 10 rows required)")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        problem_type: ProblemType
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """Prepara los datos para el entrenamiento"""
        
        # Para clustering, no hay variable objetivo
        if problem_type == ProblemType.CLUSTERING:
            X = df.copy()
            y = None
            feature_names = X.columns.tolist()
        else:
            # Separar features y target
            X = df.drop(columns=[target_column])
            y = df[target_column].copy()
            feature_names = X.columns.tolist()
            
            # Codificar variables categóricas en features
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            # Codificar target para clasificación
            if problem_type == ProblemType.CLASSIFICATION:
                if y.dtype == 'object' or y.dtype.name == 'category':
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y.astype(str)), name=target_column)
                    self.label_encoders[target_column] = le
        
        # Convertir todo a numérico
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Validar que no quedaron NaN después de la conversión
        if X.isnull().any().any():
            raise DatasetValidationError("Failed to convert all features to numeric")
        
        return X, y, feature_names
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Divide los datos en entrenamiento y prueba"""
        
        if y is None:
            # Para clustering, no hay split con target
            return X.values, X.values, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y.values,
            test_size=self.settings.TEST_SIZE,
            random_state=self.settings.RANDOM_STATE
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_prediction_data(
        self, 
        data: List[Dict[str, Any]], 
        feature_names: List[str]
    ) -> np.ndarray:
        """Prepara datos para predicción"""
        
        df = self.json_to_dataframe(data)
        
        # Validar que tenga las mismas columnas
        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
            raise DatasetValidationError(f"Missing features: {missing_cols}")
        
        # Ordenar columnas según el modelo
        df = df[feature_names]
        
        # Aplicar encoders si existen
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Convertir a numérico
        df = df.apply(pd.to_numeric, errors='coerce')
        
        if df.isnull().any().any():
            raise DatasetValidationError("Failed to convert prediction data to numeric")
        
        return df.values