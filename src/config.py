from pydantic_settings import BaseSettings, SettingsConfigDict as ConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Configuración centralizada de la aplicación"""
    
    # API Config
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "ML Training API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API REST para entrenamiento y predicción de modelos de Machine Learning"
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    # Storage
    MODEL_STORAGE_PATH: Path = Path("data/models")
    
    # ML Config
    MAX_DATASET_ROWS: int = 100000
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )


@lru_cache()
def get_settings() -> Settings:
    """Singleton de configuración"""
    return Settings()


# Crear directorio de almacenamiento si no existe
get_settings().MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)