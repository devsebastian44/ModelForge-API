"""Tests para endpoints de entrenamiento"""
import pytest
from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


def test_train_classification_model():
    """Test entrenamiento de clasificación"""
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
    
    assert response.status_code == 201
    data = response.json()
    assert "model_id" in data
    assert data["status"] == "success"
    assert "metrics" in data


def test_train_regression_model():
    """Test entrenamiento de regresión"""
    payload = {
        "dataset": [
            {"x": 1, "y": 2.5},
            {"x": 2, "y": 5.0},
            {"x": 3, "y": 7.5},
            {"x": 4, "y": 10.0},
            {"x": 5, "y": 12.5},
            {"x": 6, "y": 15.0},
            {"x": 7, "y": 17.5},
            {"x": 8, "y": 20.0},
            {"x": 9, "y": 22.5},
            {"x": 10, "y": 25.0},
        ],
        "target_column": "y",
        "problem_type": "regression",
        "algorithm": "linear_regression"
    }
    
    response = client.post("/api/v1/training", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert "model_id" in data
    assert data["metrics"]["rmse"] is not None


def test_train_clustering_model():
    """Test entrenamiento de clustering"""
    payload = {
        "dataset": [
            {"x": 1, "y": 2},
            {"x": 1.5, "y": 1.8},
            {"x": 5, "y": 8},
            {"x": 8, "y": 8},
            {"x": 1, "y": 0.6},
            {"x": 9, "y": 11},
            {"x": 8, "y": 2},
            {"x": 10, "y": 2},
            {"x": 9, "y": 3},
            {"x": 10, "y": 1},
        ],
        "problem_type": "clustering",
        "algorithm": "kmeans",
        "hyperparameters": {"n_clusters": 3}
    }
    
    response = client.post("/api/v1/training", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert "model_id" in data


def test_invalid_dataset():
    """Test con dataset inválido"""
    payload = {
        "dataset": [],
        "target_column": "target",
        "problem_type": "classification",
        "algorithm": "logistic_regression"
    }
    
    response = client.post("/api/v1/training", json=payload)
    
    assert response.status_code == 422
