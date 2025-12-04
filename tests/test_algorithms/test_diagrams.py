"""Test del módulo diagrams"""
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from visualization.diagrams import analyze_and_visualize


def list_pseudocode_files(directory: str) -> list:
    """Lista los archivos .txt en la carpeta de pseudocódigo."""
    return [f for f in os.listdir(directory) if f.endswith(".txt")]


def select_pseudocode_file(files: list) -> str:
    """Muestra un menú para seleccionar un archivo de pseudocódigo."""
    print("Seleccione un archivo de pseudocódigo:")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {file}")
    while True:
        try:
            choice = int(input("Ingrese el número del archivo: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("Selección inválida. Intente de nuevo.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número.")


def test_python_code():
    """Test con código Python"""
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(i)
"""
    result = analyze_and_visualize(code, code_type="python", output_dir="visualization/test_output")
    
    assert result["total_nodes"] > 0
    assert len(result["files"]) > 0
    assert "fibonacci" in result["recursive_functions"]
    print("✓ Test Python pasado")
    print(f"  Total nodos: {result['total_nodes']}")
    print(f"  Archivos generados: {len(result['files'])}")


def test_pseudocode(directory: str, output_base_dir: str):
    """Genera diagramas para un archivo de pseudocódigo seleccionado."""
    files = list_pseudocode_files(directory)
    if not files:
        print("No se encontraron archivos de pseudocódigo en la carpeta especificada.")
        return

    selected_file = select_pseudocode_file(files)
    pseudocode_path = os.path.join(directory, selected_file)

    with open(pseudocode_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Crear una carpeta específica para el algoritmo
    algorithm_name = os.path.splitext(selected_file)[0]
    output_dir = os.path.join(output_base_dir, algorithm_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generando diagramas para '{selected_file}' en '{output_dir}'...")
    result = analyze_and_visualize(code, code_type="pseudocode", output_dir=output_dir)

    if "error" in result:
        print(f"Error al analizar el pseudocódigo: {result['error']}")
    else:
        print("Diagramas generados con éxito.")
        print(f"Archivos generados: {result['files']}")


if __name__ == "__main__":
    pseudocode_dir = "tests/pseudocode"
    output_dir = "visualization/test_output"

    test_python_code()
    test_pseudocode(pseudocode_dir, output_dir)
    print("\n Todos los tests pasaron")