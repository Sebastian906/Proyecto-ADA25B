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
    code: str
    code_type: str = "python"
    algorithm_name: str = ""  # Opcional: nombre personalizado

@app.get("/")
def root():
    return {"message": "Analizador de Complejidades - Sistema Activo"}

@app.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    """Analiza código y genera diagramas AST organizados por algoritmo"""
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
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)