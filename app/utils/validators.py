"""Validadores personalizados"""
from typing import Any, Dict, List
from app.models.schemas import ProblemType, Algorithm


def validate_algorithm_compatibility(
    problem_type: ProblemType, 
    algorithm: Algorithm
) -> bool:
    """
    Valida que el algoritmo sea compatible con el tipo de problema.
    
    Args:
        problem_type: Tipo de problema ML
        algorithm: Algoritmo seleccionado
    
    Returns:
        True si es compatible, False en caso contrario
    """
    compatibility_map = {
        ProblemType.REGRESSION: [
            Algorithm.LINEAR_REGRESSION,
            Algorithm.RANDOM_FOREST
        ],
        ProblemType.CLASSIFICATION: [
            Algorithm.LOGISTIC_REGRESSION,
            Algorithm.RANDOM_FOREST
        ],
        ProblemType.CLUSTERING: [
            Algorithm.KMEANS
        ]
    }
    
    return algorithm in compatibility_map.get(problem_type, [])


def validate_hyperparameters(
    algorithm: Algorithm,
    hyperparameters: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Valida que los hiperparámetros sean apropiados para el algoritmo.
    
    Args:
        algorithm: Algoritmo seleccionado
        hyperparameters: Diccionario de hiperparámetros
    
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    valid_params = {
        Algorithm.LINEAR_REGRESSION: [],
        Algorithm.LOGISTIC_REGRESSION: ['C', 'max_iter', 'solver'],
        Algorithm.RANDOM_FOREST: ['n_estimators', 'max_depth', 'min_samples_split'],
        Algorithm.KMEANS: ['n_clusters', 'max_iter', 'n_init']
    }
    
    allowed = valid_params.get(algorithm, [])
    
    for param in hyperparameters.keys():
        if param not in allowed and param != 'random_state':
            return False, f"Invalid hyperparameter '{param}' for {algorithm}"
    
    return True, ""