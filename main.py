"""
Archivo `main.py`

Este archivo configura una API utilizando FastAPI para analizar pseudocódigo o código fuente, calcular complejidades, 
detectar patrones algorítmicos, y generar diagramas visuales. 

Endpoints principales:
    - GET `/`: Verifica el estado del sistema.
    - POST `/analyze`: Analiza código y genera diagramas AST organizados por algoritmo.

Dependencias:
    - `core.parser`: Para analizar pseudocódigo.
    - `core.complexity`: Para calcular complejidades.
    - `visualization.diagrams`: Para generar diagramas visuales.
    - `llm.integration`: Para integrar con Gemini y generar respuestas heurísticas.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.parser import PseudocodeParser
from core.complexity import ComplexityResult
import networkx as nx
from visualization.diagrams import analyze_and_visualize, extract_algorithm_name
from llm.integration import ask_gemini
import re

app = FastAPI(title="Analizador de Complejidades", version="1.0.0")

class CodeAnalysisRequest(BaseModel):
    """
    Modelo de datos para solicitudes de análisis de código.

    Atributos:
        code (str): Código fuente o pseudocódigo a analizar.
        code_type (str): Tipo de código ("python" o "pseudocode"). Por defecto, "python".
        algorithm_name (str): Nombre personalizado del algoritmo. Opcional.
    """
    code: str
    code_type: str = "python"
    algorithm_name: str = ""  # Opcional: nombre personalizado

@app.get("/")
def root():
    """
    Verifica el estado del sistema.

    Returns:
        Dict[str, str]: Mensaje indicando que el sistema está activo.
    """
    return {"message": "Analizador de Complejidades - Sistema Activo"}

@app.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    """
    Analiza código y genera diagramas AST organizados por algoritmo.

    Args:
        request (CodeAnalysisRequest): Solicitud con el código, tipo de código, y nombre del algoritmo.

    Returns:
        Dict[str, Any]: Resultados del análisis, incluyendo:
            - algorithm_name (str): Nombre del algoritmo analizado.
            - output_directory (str): Directorio donde se guardaron los diagramas.
            - Otros resultados generados por `analyze_and_visualize`.
    """
    try:
        # Si no proporciona nombre, extrae del código
        algo_name = request.algorithm_name or extract_algorithm_name(request.code)
        
        # Crear carpeta con el nombre del algoritmo
        output_dir = f"visualization/output/{algo_name}"
        
        result = analyze_and_visualize(
            code=request.code,
            code_type=request.code_type,
            output_dir=output_dir
        )
        
        return {
            **result,
            "algorithm_name": algo_name,
            "output_directory": output_dir
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    """
    Inicia el servidor FastAPI utilizando Uvicorn.

    Configuración:
        host: 127.0.0.1
        port: 8000
    """
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)