import sys
import os

# Agregar la carpeta raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm.integration import ask_gemini

def test_gemini_connection():
    prompt = "Explica la complejidad del algoritmo de búsqueda binaria en notación Big-O."
    response = ask_gemini(prompt)
    print("\n--- Respuesta de Gemini ---\n")
    print(response)

if __name__ == "__main__":
    test_gemini_connection()
