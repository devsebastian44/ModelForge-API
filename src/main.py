from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from src.core.config import get_settings
from src.models.schemas import HealthResponse
from src.core.exceptions import MLAPIException

# Importar routers
from src.api.v1.endpoints import training, prediction, inventory


# Configuración
settings = get_settings()

# Crear aplicación
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(MLAPIException)
async def ml_exception_handler(request: Request, exc: MLAPIException):
    """Handler para excepciones personalizadas"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para excepciones generales"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Health check
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Verifica el estado de la API.
    
    Returns:
        Estado de salud del servicio
    """
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        timestamp=datetime.now()
    )


# Incluir routers
app.include_router(
    training.router,
    prefix=settings.API_V1_PREFIX
)

app.include_router(
    prediction.router,
    prefix=settings.API_V1_PREFIX
)

app.include_router(
    inventory.router,
    prefix=settings.API_V1_PREFIX
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raíz de la API.
    """
    return {
        "message": "ML Training API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",  # nosec B104
        port=8000,
        reload=True
    )
