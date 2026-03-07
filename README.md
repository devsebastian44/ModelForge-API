# 🤖 ML Training API

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![GitLab](https://img.shields.io/badge/GitLab-Repository-orange?logo=gitlab)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)
![API](https://img.shields.io/badge/API-REST-blue)

API REST profesional para entrenamiento y predicción de modelos de Machine Learning usando FastAPI y scikit-learn.

## 🚀 Características

- ✅ **Tres tipos de problemas ML**: Regresión, Clasificación, Clustering
- ✅ **Cuatro algoritmos**: Linear Regression, Logistic Regression, Random Forest, K-Means
- ✅ **Métricas completas**: Accuracy, Precision, Recall, F1-Score, RMSE, R²
- ✅ **API REST completa** con validación Pydantic
- ✅ **Documentación automática** con Swagger/ReDoc
- ✅ **Arquitectura limpia** con separación de responsabilidades
- ✅ **Persistencia de modelos** para reutilización
- ✅ **Preparado para producción** y escalado horizontal

## 📋 Requisitos

- Python 3.9+
- pip

## 🛠️ Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd ml-api

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual

# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Copiar archivo de configuración
cp .env.example .env
```

## 🏃 Ejecución

### Modo Desarrollo
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Modo Producción
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📖 Documentación

Una vez iniciado el servidor, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔌 Endpoints Principales

### 1. Entrenar Modelo
```http
POST /api/v1/training
```

### 2. Realizar Predicción
```http
POST /api/v1/prediction
```

### 3. Listar Modelos
```http
GET /api/v1/models
```

### 4. Obtener Info de Modelo
```http
GET /api/v1/models/{model_id}
```

### 5. Eliminar Modelo
```http
DELETE /api/v1/models/{model_id}
```

## 📝 Ejemplos de Uso

### Clasificación con Random Forest

```bash
curl -X POST "http://localhost:8000/api/v1/training" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {"feature1": 1.2, "feature2": 3.4, "target": 0},
      {"feature1": 2.1, "feature2": 4.5, "target": 1}
    ],
    "target_column": "target",
    "problem_type": "classification",
    "algorithm": "random_forest",
    "hyperparameters": {"n_estimators": 100}
  }'
```

### Predicción

```bash
curl -X POST "http://localhost:8000/api/v1/prediction" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_20240125_123456",
    "data": [
      {"feature1": 1.5, "feature2": 3.2}
    ]
  }'
```

## 🏗️ Arquitectura

```
ml-api/
├── app/
│   ├── main.py              # Entry point
│   ├── config.py            # Configuración
│   ├── api/                 # Endpoints
│   │   └── v1/endpoints/
│   ├── models/              # Schemas Pydantic
│   ├── services/            # Lógica de negocio
│   │   ├── ml_service.py
│   │   ├── data_processor.py
│   │   ├── model_manager.py
│   │   └── metrics_service.py
│   └── core/                # Excepciones, utilidades
└── storage/models/          # Modelos entrenados
```

## 🧪 Testing

```bash
# Instalar dependencias de testing
pip install pytest pytest-cov httpx

# Ejecutar tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=app --cov-report=html
```

## 🔒 Seguridad

- Validación de datos con Pydantic
- Límite de tamaño de datasets
- Manejo de excepciones robusto
- CORS configurable

## 📊 Algoritmos Soportados

### Regresión
- **Linear Regression**: Para relaciones lineales
- **Random Forest Regressor**: Para relaciones no lineales

### Clasificación
- **Logistic Regression**: Clasificación binaria/multiclase
- **Random Forest Classifier**: Clasificación robusta

### Clustering
- **K-Means**: Agrupamiento no supervisado