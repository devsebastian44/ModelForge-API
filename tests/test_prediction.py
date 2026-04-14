"""Tests para endpoints de predicción"""
import pytest
from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


@pytest.fixture
def trained_model_id():
    """Fixture que entrena un modelo y retorna su ID"""
    payload = {
        "dataset": [
            {"feature1": 1.2, "feature2": 3.4, "target": 0},
            {"feature1": 2.1, "feature2": 4.5, "target": 1},
            {"feature1": 1.5, "feature2": 3.8, "target": 0},
            {"feature1": 2.3, "feature2": 4.2, "target": 1},
            {"feature1": 1.8, "feature2": 3.6, "target": 0},
            {"feature1": 2.5, "feature2": 4.8, "target": 1},
            {"feature1": 1.3, "feature2": 3.2, "target": 0},
            {"feature1": 2.2, "feature2": 4.4, "target": 1},
            {"feature1": 1.6, "feature2": 3.7, "target": 0},
            {"feature1": 2.4, "feature2": 4.6, "target": 1},
        ],
        "target_column": "target",
        "problem_type": "classification",
        "algorithm": "logistic_regression"
    }
    
    response = client.post("/api/v1/training", json=payload)
    return response.json()["model_id"]


def test_predict_success(trained_model_id):
    """Test predicción exitosa"""
    payload = {
        "model_id": trained_model_id,
        "data": [
            {"feature1": 1.4, "feature2": 3.5},
            {"feature1": 2.2, "feature2": 4.3}
        ]
    }
    
    response = client.post("/api/v1/prediction", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["status"] == "success"


def test_predict_model_not_found():
    """Test con modelo inexistente"""
    payload = {
        "model_id": "model_inexistente",
        "data": [
            {"feature1": 1.4, "feature2": 3.5}
        ]
    }
    
    response = client.post("/api/v1/prediction", json=payload)
    
    assert response.status_code == 404


def test_predict_empty_data():
    """Test con datos vacíos"""
    payload = {
        "model_id": "model_test",
        "data": []
    }
    
    response = client.post("/api/v1/prediction", json=payload)
    
    assert response.status_code == 422
