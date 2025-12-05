"""
Archivo `main.py`

Este archivo configura una API utilizando FastAPI para analizar pseudocódigo o código fuente, calcular complejidades, 
detectar patrones algorítmicos, y generar diagramas visuales. También incluye un menú interactivo para ejecutar 
funcionalidades específicas desde la terminal.
"""

from pathlib import Path
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.parser import PseudocodeParser
from core.complexity import ComplexityResult, ComplexityAnalyzer
from visualization.diagrams import analyze_and_visualize, extract_algorithm_name
from llm.integration import ask_gemini
from llm.validation import validate_efficiency
import uvicorn
import re
import socket
import sys

app = FastAPI(title="Analizador de Complejidades", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",  # Live Server (VS Code)
        "http://127.0.0.1:5500",
        "http://localhost:8080",  # http-server
        "http://127.0.0.1:8080",
        "file://",  # Para abrir HTML directamente
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear directorio si no existe
visualization_dir = Path("visualization/output")
visualization_dir.mkdir(parents=True, exist_ok=True)

# Montar directorio como estático
app.mount(
    "/visualization/output",
    StaticFiles(directory="visualization/output"),
    name="visualization"
)

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

        # Análisis principal
        print("→ Paso 1: Análisis de visualización...")
        result = analyze_and_visualize(
            code=request.code,
            code_type=request.code_type,
            output_dir=output_dir
        )
        print(f"✓ Nodos generados: {result.get('total_nodes', 0)}")

        # Análisis línea por línea
        print("\n→ Paso 2: Análisis línea por línea...")
        parser = PseudocodeParser()
        line_analysis = parser.analyze_by_line(request.code)

        # Filtrar líneas relevantes (sin summary)
        lines_detail = [
            line for line in line_analysis 
            if line.get('type') != 'summary'
        ]
        print(f"✓ Líneas analizadas: {len(lines_detail)}")

        # Extraer recurrencia
        recurrence_info = None
        for line in line_analysis:
            if line.get('type') == 'summary' and 'recurrence' in line:
                recurrence_info = {
                    'equation': line.get('recurrence'),
                    'simplified': line.get('recurrence')
                }
                print(f"✓ Recurrencia detectada: {recurrence_info['equation']}")
                break

        if not recurrence_info:
            print("⚠ No se detectó recurrencia")

        # Análisis de Complejidad
        print("\n→ Paso 3: Análisis de complejidad...")
        analyzer = ComplexityAnalyzer()
        complexity_result = analyzer.analyze(request.code)
        print(f"✓ Big-O: {complexity_result.big_o}")

        # Validación con LLM (Gemini)
        try:
            validation_result = validate_efficiency(
                request.code,
                algo_name,
                is_file=False
            )
            
            validation_summary = {
                'overall_score': validation_result.overall_efficiency_score,
                'complexity_correctness': validation_result.complexity_correctness,
                'recurrence_correctness': validation_result.recurrence_correctness,
                'mathematical_rigor': validation_result.mathematical_rigor,
                'auto_analysis': validation_result.auto_analysis,
                'gemini_analysis': validation_result.gemini_analysis[:200] + '...',  # Truncar para log
                'complexity_details': validation_result.complexity_details,
                'recurrence_details': validation_result.recurrence_details,
                'critical_errors': validation_result.critical_errors,
                'warnings': validation_result.warnings,
                'corrections': validation_result.corrections
            }
            print(f"✓ Validación completada - Score: {validation_summary['overall_score']:.1f}/100")
            
        except Exception as e:
            print(f"⚠ Error en validación con Gemini: {e}")
            validation_summary = None

        # Convertir rutas locales a URLs accesibles desde el frontend
        if "files" in result and result["files"]:
            result["files"] = [
                f"http://localhost:8000/{file}" if not file.startswith("http") else file
                for file in result["files"]
            ]

        response_data = {
            **result,
            "algorithm_name": algo_name,
            "output_directory": output_dir,
            "line_analysis": lines_detail,
            "recurrence": recurrence_info,
            "validation": validation_summary
        }

        # LOG FINAL
        print(f"✓ RESPUESTA CONSTRUIDA:")
        print(f"  - Complejidad: ✓")
        print(f"  - Patrones: {len(result.get('patterns', []))}")
        print(f"  - Líneas analizadas: {len(lines_detail)}")
        print(f"  - Recurrencia: {'✓' if recurrence_info else '✗'}")
        print(f"  - Validación: {'✓' if validation_summary else '✗'}")
        print(f"  - Archivos: {len(result.get('files', []))}")
        
        return response_data
        
    except Exception as e:
        import traceback
        print(f"\nERROR EN ANÁLISIS:")
        traceback.print_exc()
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
            # Configure uvicorn with proper socket options for Windows
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=8000,
                log_level="info"
            )
            # Enable SO_REUSEADDR to allow immediate rebinding on Windows
            server = uvicorn.Server(config)
            try:
                import asyncio
                asyncio.run(server.serve())
            except KeyboardInterrupt:
                print("\nServidor detenido.")
            except OSError as e:
                print(f"Error al iniciar servidor: {e}")
                print("Intenta de nuevo en unos segundos o usa otro puerto.")
                sys.exit(1)
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