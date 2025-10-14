"""Test del módulo diagrams"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from visualization.diagrams import analyze_and_visualize

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

def test_pseudocode():
    """Test con pseudocódigo"""
    code = """
PARA i DESDE 0 HASTA n-1 HACER
    arreglo[i] <- i + 1
FIN PARA

MIENTRAS izquierda <= derecha HACER
    medio <- (izquierda + derecha) / 2
    SI arreglo[medio] == x ENTONCES
        encontrado <- VERDADERO
    FIN SI
FIN MIENTRAS
"""
    result = analyze_and_visualize(code, code_type="pseudocode", output_dir="visualization/test_output")
    
    assert result["total_nodes"] > 0
    assert result["complexity"] is not None
    print("✓ Test Pseudocódigo pasado")
    print(f"  Total nodos: {result['total_nodes']}")
    print(f"  Complejidad: {result['complexity']}")

if __name__ == "__main__":
    test_python_code()
    test_pseudocode()
    print("\n✅ Todos los tests pasaron")