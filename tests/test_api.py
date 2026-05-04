from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

DUMMY_DATASET = [
    {"feature1": 1.2, "feature2": 3.4, "target": 0},
    {"feature1": 2.1, "feature2": 4.5, "target": 1},
    {"feature1": 3.3, "feature2": 2.2, "target": 0},
    {"feature1": 4.1, "feature2": 1.5, "target": 1},
    {"feature1": 2.5, "feature2": 2.5, "target": 0},
    {"feature1": 1.8, "feature2": 3.1, "target": 1},
    {"feature1": 3.9, "feature2": 2.8, "target": 0},
    {"feature1": 2.2, "feature2": 4.1, "target": 1},
    {"feature1": 3.1, "feature2": 1.9, "target": 0},
    {"feature1": 4.5, "feature2": 2.1, "target": 1},
]

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "ML Training API"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_full_ml_lifecycle():
    # 1. Train
    train_payload = {
        "dataset": DUMMY_DATASET,
        "target_column": "target",
        "problem_type": "classification",
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 10, "max_depth": 3}
    }
    response = client.post("/api/v1/training", json=train_payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert "model_id" in data
    model_id = data["model_id"]
    
    # 2. Predict
    pred_payload = {
        "model_id": model_id,
        "data": [
            {"feature1": 1.5, "feature2": 3.2},
            {"feature1": 3.5, "feature2": 2.0}
        ]
    }
    response = client.post("/api/v1/prediction", json=pred_payload)
    assert response.status_code == 200, response.text
    assert "predictions" in response.json()
    
    # 3. List Models
    response = client.get("/api/v1/models")
    assert response.status_code == 200, response.text
    models_data = response.json()
    assert any(m["model_id"] == model_id for m in models_data["models"])
    
    # 4. Get Model Info
    response = client.get(f"/api/v1/models/{model_id}")
    assert response.status_code == 200, response.text
    assert response.json()["model_id"] == model_id
    
    # 5. Delete Model
    response = client.delete(f"/api/v1/models/{model_id}")
    assert response.status_code == 204, response.text
