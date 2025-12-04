"""
Script de pruebas para el Analizador de Complejidad
Prueba diferentes tipos de algoritmos y valida O, Ω, Θ
"""

import sys
import os

# Agregar el directorio core al path (dos niveles arriba, luego core)
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'core')
sys.path.insert(0, core_dir)

from complexity import ComplexityAnalyzer
from patterns import PatternRecognizer
from pathlib import Path

# Helper ligero para cargar ejemplos desde tests/pseudocode
EXAMPLES_DIR = Path(__file__).parent.parent / 'pseudocode'

def load_example(name: str) -> str:
    import re
    # 1) Intentar ruta exacta
    p = EXAMPLES_DIR / name
    if not p.exists():
        p = EXAMPLES_DIR / f"{name}.txt"
    if p.exists():
        return p.read_text(encoding='utf-8')

    # 2) Buscar por tokens en el nombre del archivo
    key = Path(name).stem.lower()
    tokens = re.findall(r"\w+", key)
    candidates = []
    for f in EXAMPLES_DIR.glob('*.txt'):
        fname = f.name.lower()
        score = sum(1 for t in tokens if t in fname)
        if score > 0:
            candidates.append((score, f))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1].read_text(encoding='utf-8')

    # 3) Buscar por contenido (tokens dentro del contenido)
    content_candidates = []
    for f in EXAMPLES_DIR.glob('*.txt'):
        content = f.read_text(encoding='utf-8').lower()
        score = sum(1 for t in tokens if t in content)
        if score > 0:
            content_candidates.append((score, f))
    if content_candidates:
        content_candidates.sort(reverse=True)
        return content_candidates[0][1].read_text(encoding='utf-8')

    # Si no se encuentra, listar disponibles para ayudar al debug
    available = ', '.join([p.name for p in EXAMPLES_DIR.glob('*.txt')])
    raise FileNotFoundError(f"Ejemplo no encontrado: {name} (buscado en {EXAMPLES_DIR}). Archivos disponibles: {available}")

def list_examples():
    return sorted([p.name for p in EXAMPLES_DIR.glob('*.txt')])

def print_header(title):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_analysis(name, code, show_code=True):
    """
    Analiza y muestra resultados para un algoritmo.

    Args:
        name (str): Nombre del algoritmo.
        code (str): Pseudocódigo a analizar.
        show_code (bool): Si mostrar el código completo.
    """
    print(f"\n{'─' * 80}")
    print(f"PRUEBA: {name}")
    print(f"{'─' * 80}")
    
    if show_code:
        print("\nPseudocódigo:")
        print(code)
    
    # Análisis de recurrencia
    print("\nANÁLISIS DE RECURRENCIA:")
    recurrence_result = analyze_pseudocode(code)
    print("\nECUACIÓN DE RECURRENCIA COMPLETA:")
    print(f"  T(n) = {recurrence_result['recurrence']}")
    print("\nECUACIÓN DE RECURRENCIA SIMPLIFICADA:")
    print(f"  T(n) = {recurrence_result['simplified_recurrence']}")
    print("\nCONSTANTES:")
    for const, value in recurrence_result["constants"].items():
        print(f"  {const}: {value}")
    print(f"\nCOMPLEJIDAD ESPACIAL: {recurrence_result['space_complexity']}")

def analyze_pseudocode(code: str):
    """
    Analiza el pseudocódigo línea por línea, asigna costos y construye la ecuación de recurrencia.

    Args:
        code (str): Pseudocódigo a analizar.

    Returns:
        dict: Resultados del análisis, incluyendo la ecuación de recurrencia completa y simplificada.
    """
    lines = code.strip().split("\n")
    equation = []
    constants = {}
    space_usage = 0
    indent_level = 0

    print("\nANÁLISIS LÍNEA POR LÍNEA:")
    print("=" * 80)

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        # Identificar el tipo de línea y asignar costos
        if line.startswith("for") or line.startswith("while"):
            indent_level += 1
            constants[f"C{idx}"] = f"O(n^{indent_level})"
            equation.append(f"C{idx}*n^{indent_level}")
            print(f"{line:<50} --------------------- C{idx}*n^{indent_level}")
        elif line.startswith("if") or line.startswith("else"):
            constants[f"C{idx}"] = "O(1)"
            equation.append(f"C{idx}")
            print(f"{line:<50} --------------------- C{idx}")
        else:
            constants[f"C{idx}"] = "O(1)"
            equation.append(f"C{idx}")
            print(f"{line:<50} --------------------- C{idx}")

        # Espacio adicional si se detectan estructuras de datos
        if "array" in line or "list" in line or "matrix" in line:
            space_usage += 1

    # Construir la ecuación de recurrencia completa
    recurrence = " + ".join(equation)

    # Simplificar la ecuación de recurrencia agrupando constantes
    grouped_terms = []
    constant_count = 0  # Contador para las constantes independientes

    for term in equation:
        if "*n" in term:
            grouped_terms.append(term)  # Mantener términos dependientes de n
        else:
            constant_count += 1  # Contar las constantes independientes

    # Si hay constantes independientes, agruparlas como una sola 'C'
    if constant_count > 0:
        grouped_terms.append("C")

    simplified_equation = " + ".join(grouped_terms)

    return {
        "recurrence": recurrence,
        "simplified_recurrence": simplified_equation,
        "constants": constants,
        "space_complexity": f"O({space_usage})" if space_usage > 0 else "O(1)"
    }

def run_all_tests():
    """Muestra los archivos disponibles y permite seleccionar uno para ejecutar la prueba"""
    
    print_header("LISTA DE ARCHIVOS DISPONIBLES EN 'test/pseudocode'")
    
    # Listar los archivos disponibles en el directorio EXAMPLES_DIR
    examples = list_examples()
    if not examples:
        print("No se encontraron archivos en el directorio 'test/pseudocode'.")
        return
    
    print("Archivos disponibles:")
    for idx, example in enumerate(examples, start=1):
        print(f"  {idx}. {example}")
    
    print("\nSelecciona un archivo para analizar (ingresa el número correspondiente):")
    try:
        choice = int(input("Número: ").strip())
        if 1 <= choice <= len(examples):
            selected_file = examples[choice - 1]
            print(f"\nEjecutando prueba para: {selected_file}")
            code = load_example(selected_file)
            print_analysis(selected_file, code)
        else:
            print("Número inválido. Por favor, selecciona un número de la lista.")
    except ValueError:
        print("Entrada inválida. Por favor, ingresa un número.")

def run_custom_test():
    """Permite probar código personalizado"""
    print_header("PRUEBA PERSONALIZADA")
    print("\nIngresa tu pseudocódigo (termina con una línea vacía):")
    print("Ejemplo: for i = 0 to n do")
    print("         end")
    print()
    
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    
    if lines:
        code = "\n".join(lines)
        print_analysis("Código Personalizado", code, show_code=False)
    else:
        print("No se ingresó código.")

def run_interactive_menu():
    """Menú interactivo para ejecutar pruebas"""
    while True:
        print("\n" + "=" * 80)
        print("  MENÚ DE PRUEBAS - ANALIZADOR DE COMPLEJIDAD")
        print("=" * 80)
        print("\n1. Ejecutar pruebas (recomendado)")
        print("2. Probar código personalizado")
        print("3. Ver ejemplos específicos:")
        print("   a. Loop simple")
        print("   b. Loops anidados")
        print("   c. Búsqueda binaria")
        print("   d. Recursión")
        print("   e. Divide y conquista")
        print("4. Salir")
        print()
        
        choice = input("Selecciona una opción: ").strip().lower()
        
        if choice == "1":
            run_all_tests()
        elif choice == "2":
            run_custom_test()
        elif choice == "3a" or choice == "3 a":
            code = load_example('for_simple.txt')
            print_analysis("Loop Simple", code)
        elif choice == "3b" or choice == "3 b":
            code = load_example('nested_for.txt')
            print_analysis("Loops Anidados", code)
        elif choice == "3c" or choice == "3 c":
            code = load_example('binary_search.txt')
            print_analysis("Búsqueda Binaria", code)
        elif choice == "3d" or choice == "3 d":
            code = load_example('factorial.txt')
            print_analysis("Factorial Recursivo", code)
        elif choice == "3e" or choice == "3 e":
            code = load_example('merge_sort.txt')
            print_analysis("Merge Sort", code)
        elif choice == "4":
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción inválida. Intenta de nuevo.")

def quick_test():
    """Prueba rápida con un algoritmo de ejemplo"""
    print_header("PRUEBA RÁPIDA")
    
    code = """
    PARA i DESDE 0 HASTA n-1 HACER
        PARA j DESDE 0 HASTA n-1 HACER
            SI matriz[i][j] > 0 ENTONCES
                suma = suma + matriz[i][j]
            FIN SI
        FIN PARA
    FIN PARA
    """
    
    print_analysis("Suma de Matriz (español)", code)

if __name__ == "__main__":
    import sys
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     ANALIZADOR DE COMPLEJIDAD ALGORÍTMICA                      ║
    ║     Análisis de Big-O, Omega y Theta                           ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Si se pasa argumento 'all', ejecutar todas las pruebas
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all_tests()
    # Si se pasa 'quick', ejecutar prueba rápida
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    # Si no, mostrar menú interactivo
    else:
        print("\nOpciones de ejecución:")
        print("  python test_complexity.py        → Menú interactivo")
        print("  python test_complexity.py all    → Ejecutar todas las pruebas")
        print("  python test_complexity.py quick  → Prueba rápida")
        print()
        
        try:
            run_interactive_menu()
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")