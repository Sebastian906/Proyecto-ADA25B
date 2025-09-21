from fastapi import FastAPI
from core.parser import PseudocodeParser
from core.complexity import ComplexityAnalyzer
from llm.integration import LLMIntegration

app = FastAPI(title="Analizador de Complejidades", version="1.0.0")

@app.get("/")
def root():
    return {"message": "Analizador de Complejidades - Sistema Activo"}

__name__ == "__main__"
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)