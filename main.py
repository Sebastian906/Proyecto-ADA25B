"""
Archivo `main.py`

Este archivo configura una API utilizando FastAPI para analizar pseudocódigo o código fuente, calcular complejidades, 
detectar patrones algorítmicos, y generar diagramas visuales. También incluye un menú interactivo para ejecutar 
funcionalidades específicas desde la terminal.
"""

import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.parser import PseudocodeParser
from core.complexity import ComplexityResult
from visualization.diagrams import analyze_and_visualize, extract_algorithm_name
from llm.integration import ask_gemini
import uvicorn
import re

app = FastAPI(title="Analizador de Complejidades", version="1.0.0")

class CodeAnalysisRequest(BaseModel):
    code: str
    code_type: str = "python"
    algorithm_name: str = ""

@app.get("/")
def root():
    return {"message": "Analizador de Complejidades - Sistema Activo"}

@app.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    try:
        algo_name = request.algorithm_name or extract_algorithm_name(request.code)
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

def execute_command(command):
    """
    Ejecuta un comando en la terminal y permite interacción directa.
    """
    try:
        subprocess.run(
            command,
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")

def menu():
    """
    Menú interactivo para ejecutar funcionalidades específicas del sistema.
    """
    while True:
        print("\n--- Menú Principal ---")
        print("1. Iniciar servidor FastAPI")
        print("2. Verificar funcionalidad de LLM (Gemini)")
        print("3. Validar un algoritmo")
        print("4. Validar concordancia entre el sistema y el LLM")
        print("5. Identificar patrones algorítmicos")
        print("6. Calcular la complejidad de un algoritmo")
        print("7. Crear diagramas de algoritmos")
        print("0. Salir")
        
        choice = input("Seleccione una opción: ")
        
        if choice == "1":
            print("Iniciando servidor FastAPI...")
            uvicorn.run(app, host="127.0.0.1", port=8000)
        elif choice == "2":
            print("Ejecutando verificación de funcionalidad de LLM (Gemini)...")
            execute_command("python -m tests.test_algorithms.test_llm")
        elif choice == "3":
            print("Ejecutando validación de un algoritmo...")
            execute_command("python -m tests.test_algorithms.test_validation")
        elif choice == "4":
            print("Ejecutando validación de concordancia entre el sistema y el LLM...")
            execute_command("python -m tests.test_algorithms.diagnostic_validation")
        elif choice == "5":
            print("Ejecutando identificación de patrones algorítmicos...")
            execute_command("python -m tests.test_algorithms.test_patterns")
        elif choice == "6":
            print("Ejecutando cálculo de complejidad de un algoritmo...")
            execute_command("python -m tests.test_algorithms.test_complexity")
        elif choice == "7":
            print("Ejecutando creación de diagramas de algoritmos...")
            execute_command("python -m tests.test_algorithms.test_diagrams")
        elif choice == "0":
            print("Saliendo del sistema. ¡Hasta luego!")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    print("Bienvenido al Analizador de Complejidades.")
    menu()