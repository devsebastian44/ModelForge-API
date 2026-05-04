# ModelForge API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Engine-F7931E?style=flat&logo=scikitlearn)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat&logo=pydantic)
![License](https://img.shields.io/badge/License-MIT-green?style=flat&logo=opensourceinitiative)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=flat&logo=github-actions&logoColor=white)

---

## 🧠 Overview

ModelForge API es una API REST de Machine Learning construida sobre **FastAPI** y **scikit-learn**,
diseñada para exponer capacidades de entrenamiento y predicción de modelos ML a través de
endpoints HTTP con validación estricta de datos mediante **Pydantic**.

El proyecto implementa una arquitectura de servicios en capas donde la lógica de negocio de ML
— preprocesamiento de datos, selección de algoritmos, cálculo de métricas y persistencia de
modelos — está completamente desacoplada de la capa de transporte HTTP. Los modelos entrenados
se serializan con **joblib** y se persisten en disco dentro de `storage/models/`, permitiendo
reutilizarlos en predicciones posteriores sin necesidad de reentrenamiento.

Este proyecto parece orientado a servir como backend de ML reutilizable, integrándose con
sistemas de datos externos o frontends analíticos mediante su API versionada (`/api/v1`), y
cuenta con documentación interactiva generada automáticamente por FastAPI (Swagger UI y ReDoc).

---

## ⚙️ Features

- **Entrenamiento on-demand**: envía un dataset en formato JSON con el tipo de problema y
  algoritmo deseado, y la API entrena, evalúa y persiste el modelo automáticamente.
- **Predicción sobre modelos guardados**: realiza inferencia sobre cualquier modelo previamente
  entrenado referenciándolo por su `model_id`.
- **Gestión de modelos**: listado, consulta de metadatos y eliminación de modelos almacenados.
- **Cinco algoritmos ML**: Linear Regression, Logistic Regression, Random Forest Regressor,
  Random Forest Classifier y K-Means Clustering.
- **Tres tipos de problemas soportados**: Regresión, Clasificación y Clustering no supervisado.
- **Métricas completas por tipo de problema**: Accuracy, Precision, Recall, F1-Score (clasif.),
  RMSE y R² (regresión).
- **Validación estricta de entradas** con esquemas Pydantic en cada endpoint.
- **Documentación automática** disponible en `/docs` (Swagger UI) y `/redoc` (ReDoc).
- **Health check** en `/health` para monitoreo de disponibilidad.
- **CORS configurable** desde variables de entorno para integración con frontends.
- **Multipart/form-data** soportado via `python-multipart` para potencial carga de archivos.
- **Configuración por entorno** con `pydantic-settings` y `python-dotenv`.

---

## 🛠️ Tech Stack

| Capa | Tecnología |
|---|---|
| Framework HTTP | FastAPI |
| Servidor ASGI | Uvicorn (con extras `[standard]`: websockets, HTTP/2) |
| Validación / Schemas | Pydantic v2 + pydantic-settings |
| Motor ML | scikit-learn |
| Manipulación de datos | pandas, numpy |
| Persistencia de modelos | joblib |
| Carga de archivos | python-multipart |
| Variables de entorno | python-dotenv |
| Testing | pytest + pytest-cov + httpx |
| Lenguaje | Python 3.9+ |

---

## 📦 Installation

### Prerrequisitos

- Python `>=3.9`
- pip

### Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/devsebastian44/ModelForge-API.git
cd ModelForge-API

# 2. Crear y activar entorno virtual
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env según tu entorno
```

---

## ▶️ Usage

### Modo desarrollo (con auto-reload)

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Modo producción (multi-worker)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Una vez iniciado el servidor, accede a:

- **Swagger UI** (interactivo): `http://localhost:8000/docs`
- **ReDoc** (documentación): `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/training` | Entrenar un nuevo modelo ML |
| `POST` | `/api/v1/prediction` | Realizar predicción con un modelo guardado |
| `GET` | `/api/v1/models` | Listar todos los modelos entrenados |
| `GET` | `/api/v1/models/{model_id}` | Obtener metadatos de un modelo específico |
| `DELETE` | `/api/v1/models/{model_id}` | Eliminar un modelo persistido |
| `GET` | `/health` | Estado del servicio |

### Ejemplo: Entrenar un clasificador Random Forest

```bash
curl -X POST "http://localhost:8000/api/v1/training" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {"feature1": 1.2, "feature2": 3.4, "target": 0},
      {"feature1": 2.1, "feature2": 4.5, "target": 1},
      {"feature1": 3.3, "feature2": 2.2, "target": 0}
    ],
    "target_column": "target",
    "problem_type": "classification",
    "algorithm": "random_forest",
    "hyperparameters": {"n_estimators": 100, "max_depth": 5}
  }'
```

### Ejemplo: Predecir con un modelo entrenado

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

### Ejecutar tests

```bash
# Instalar dependencias de testing
pip install pytest pytest-cov httpx

# Ejecutar suite completa
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
ModelForge-API/
│
├── src/                          # Paquete principal de la aplicación
│   ├── main.py                   # Entry point FastAPI: instancia de app, CORS, routers
│   ├── config.py                 # Configuración centralizada con pydantic-settings
│   │
│   ├── api/                      # Capa de transporte HTTP
│   │   └── v1/
│   │       └── endpoints/        # Routers FastAPI por dominio
│   │           ├── training.py   # Endpoint POST /training
│   │           ├── prediction.py # Endpoint POST /prediction
│   │           └── models.py     # Endpoints GET/DELETE /models
│   │
│   ├── models/                   # Schemas Pydantic (contratos de la API)
│   │   ├── training.py           # TrainingRequest, TrainingResponse
│   │   ├── prediction.py         # PredictionRequest, PredictionResponse
│   │   └── model_info.py         # ModelInfo, ModelListResponse
│   │
│   ├── services/                 # Capa de lógica de negocio ML
│   │   ├── ml_service.py         # Orquestador principal: entrena y predice
│   │   ├── data_processor.py     # Preprocesamiento y validación del dataset
│   │   ├── model_manager.py      # Persistencia/carga de modelos con joblib
│   │   └── metrics_service.py    # Cálculo de métricas por tipo de problema
│   │
│   └── core/                     # Utilidades transversales
│       └── exceptions.py         # Excepciones personalizadas y manejadores HTTP
│
├── tests/                        # Suite de tests
│   └── test_api.py               # Tests de integración con httpx TestClient
│
├── storage/
│   └── models/                   # Almacenamiento de modelos serializados (.joblib)
│
├── .env.example                  # Plantilla de variables de entorno
├── requirements.txt              # Dependencias de producción
└── LICENSE                       # Licencia MIT
```

---

## 🔐 Security

- **Validación de esquemas con Pydantic v2**: todas las entradas de la API son validadas
  estrictamente contra schemas tipados antes de llegar a la lógica de negocio, rechazando
  automáticamente datos malformados o con tipos incorrectos.
- **Manejo robusto de excepciones**: excepciones personalizadas en `core/exceptions.py`
  previenen que errores internos del motor ML expongan stack traces al cliente.
- **CORS configurable por entorno**: los orígenes permitidos se gestionan desde variables de
  entorno, evitando configuraciones permisivas hardcodeadas en producción.
- **Límite de tamaño de dataset**: la capa de servicio implementa restricciones sobre el
  volumen de datos aceptado por request, previniendo ataques de denegación de servicio por
  payloads excesivamente grandes.
- **Sin almacenamiento de datos sensibles**: el sistema solo persiste modelos serializados
  (`.joblib`), no los datasets de entrenamiento originales.
- **Separación de entornos via `.env`**: las configuraciones sensibles de producción se
  mantienen fuera del código fuente mediante `python-dotenv` y `pydantic-settings`.

> Para despliegues en producción se recomienda añadir autenticación mediante API keys o
> JWT, configurar HTTPS y limitar los orígenes CORS al dominio del cliente.

---

## 🚀 Roadmap

Basado en la arquitectura actual detectada en el código, estas son las evoluciones técnicas
sugeridas:

- [ ] **Autenticación y autorización**: implementar API keys o JWT para proteger endpoints
  en producción.
- [ ] **Base de datos de metadatos**: migrar el registro de modelos de sistema de archivos a
  SQLite/PostgreSQL con SQLAlchemy para búsquedas y filtros avanzados.
- [ ] **Entrenamiento asíncrono**: usar `BackgroundTasks` de FastAPI o Celery para entrenar
  modelos pesados sin bloquear la respuesta HTTP.
- [ ] **Soporte para más algoritmos**: SVM, XGBoost, LightGBM, redes neuronales via
  scikit-learn compatible API.
- [ ] **Versionado de modelos**: permitir múltiples versiones de un mismo modelo y
  comparación de métricas entre versiones.
- [ ] **Carga de datasets por archivo**: habilitar ingesta de CSV/Parquet directamente via
  multipart (ya soportado por `python-multipart`).
- [ ] **Containerización con Docker**: Dockerfile y docker-compose para despliegue
  reproducible.
- [ ] **Pipeline CI/CD**: automatización de tests, linting y despliegue en cada push.
- [ ] **Soporte MLflow**: integración para tracking de experimentos, comparación de runs y
  registro centralizado de modelos.
- [ ] **Dashboard web**: interfaz visual para entrenar, explorar y comparar modelos sin
  necesidad de curl o clientes API.
- [ ] **Límite de rate**: middleware de rate-limiting para ambientes multi-tenant.
- [ ] **Exportación de modelos**: endpoints para descargar modelos en formato ONNX o pickle
  para uso externo.

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Ethical Usage & Disclaimer

> [!IMPORTANT]
> **This project is for educational and ethical cybersecurity purposes only.**
> 
> The code, datasets, and machine learning models in this repository are provided "as-is" for learning, research, and portfolio demonstration. Users must ensure that any usage of this software complies with applicable local, state, and federal laws. The authors assume no liability and are not responsible for any misuse or damage caused by this program.

---

## 📄 License

Este proyecto está bajo la licencia **MIT**.

> Licencia detectada directamente desde el archivo `LICENSE` en la raíz del repositorio.

---

## 👨‍💻 Author

**Sebastian Zhunaula** — [@devsebastian44](https://github.com/devsebastian44)

Desarrollador full-stack e ingeniero de datos con enfoque en APIs de Machine Learning,
arquitecturas orientadas a servicios y buenas prácticas de ingeniería de software aplicada
a proyectos de IA.