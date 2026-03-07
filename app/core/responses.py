"""Response builders personalizados"""
from typing import Any, Dict
from datetime import datetime


def success_response(
    data: Any,
    message: str = "Success"
) -> Dict[str, Any]:
    """
    Construye una respuesta exitosa estandarizada.
    
    Args:
        data: Datos a retornar
        message: Mensaje de éxito
    
    Returns:
        Diccionario con respuesta formateada
    """
    return {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }


def error_response(
    error: str,
    details: Any = None
) -> Dict[str, Any]:
    """
    Construye una respuesta de error estandarizada.
    
    Args:
        error: Mensaje de error
        details: Detalles adicionales del error
    
    Returns:
        Diccionario con error formateado
    """
    response = {
        "status": "error",
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        response["details"] = details
    
    return response