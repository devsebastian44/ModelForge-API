import joblib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.models.schemas import (
    ProblemType, Algorithm, MetricsResponse, ModelInfo
)
from src.core.exceptions import ModelNotFoundError
from src.config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestor de modelos entrenados"""
    
    def __init__(self):
        self.settings = get_settings()
        self.storage_path = self.settings.MODEL_STORAGE_PATH
    
    def generate_model_id(self) -> str:
        """Genera un ID único para el modelo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"model_{timestamp}"
    
    def save_model(
        self,
        model_id: str,
        model: Any,
        problem_type: ProblemType,
        algorithm: Algorithm,
        feature_names: List[str],
        metrics: MetricsResponse,
        label_encoders: Optional[Dict] = None
    ) -> None:
        """Guarda un modelo entrenado"""
        
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Guardar el modelo
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Guardar encoders si existen
        if label_encoders:
            encoders_path = model_dir / "encoders.pkl"
            joblib.dump(label_encoders, encoders_path)
        
        # Guardar metadata
        metadata = {
            "model_id": model_id,
            "problem_type": problem_type.value,
            "algorithm": algorithm.value,
            "feature_names": feature_names,
            "metrics": metrics.model_dump(),
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, model_id: str) -> tuple[Any, Dict, Optional[Dict]]:
        """Carga un modelo entrenado"""
        
        model_dir = self.storage_path / model_id
        
        if not model_dir.exists():
            raise ModelNotFoundError(model_id)
        
        # Cargar modelo
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise ModelNotFoundError(model_id)
        
        model = joblib.load(model_path)
        
        # Cargar metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cargar encoders si existen
        encoders = None
        encoders_path = model_dir / "encoders.pkl"
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)
        
        return model, metadata, encoders
    
    def list_models(self) -> List[ModelInfo]:
        """Lista todos los modelos guardados"""
        
        models = []
        
        for model_dir in self.storage_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_info = ModelInfo(
                    model_id=metadata["model_id"],
                    problem_type=ProblemType(metadata["problem_type"]),
                    algorithm=Algorithm(metadata["algorithm"]),
                    feature_names=metadata["feature_names"],
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                    metrics=MetricsResponse(**metadata["metrics"])
                )
                
                models.append(model_info)
            except Exception as e:
                logger.warning(f"Metadata corrupted in {model_dir.name}: {e}")
                continue
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo"""
        
        model_dir = self.storage_path / model_id
        
        if not model_dir.exists():
            raise ModelNotFoundError(model_id)
        
        # Eliminar todos los archivos del modelo
        for file_path in model_dir.iterdir():
            file_path.unlink()
        
        model_dir.rmdir()
        return True
