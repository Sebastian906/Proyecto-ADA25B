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
    Analiza y muestra resultados para un algoritmo
    
    Args:
        name (str): Nombre del algoritmo
        code (str): Pseudocódigo a analizar
        show_code (bool): Si mostrar el código completo
    """
    print(f"\n{'─' * 80}")
    print(f"PRUEBA: {name}")
    print(f"{'─' * 80}")
    
    if show_code:
        print("\nPseudocódigo:")
        print(code)
    
    # Análisis de complejidad
    print("\nANÁLISIS DE COMPLEJIDAD:")
    analyzer = ComplexityAnalyzer()
    result = analyzer.analyze(code)
    
    print(f"  • Big-O (Peor caso):     {result.big_o}")
    print(f"  • Omega (Mejor caso):    {result.omega}")
    print(f"  • Theta (Caso promedio): {result.theta}")
    print(f"\n  Explicación: {result.explanation}")
    
    if result.recurrence:
        print(f"  Recurrencia: {result.recurrence}")
    if result.pattern_type:
        print(f"  Patrón: {result.pattern_type}")
    
    # Análisis de patrones
    print("\nPATRONES DETECTADOS:")
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    
    if patterns:
        for pattern in patterns:
            print(f"  • {pattern['description']}")
            print(f"    Complejidad esperada: {pattern['complexity']}")
            print(f"    Confianza: {pattern['confidence']}")
    else:
        print("  No se detectaron patrones especiales")
    
    print()

def run_all_tests():
    """Ejecuta todas las pruebas de ejemplo"""
    
    print_header("SUITE DE PRUEBAS - ANALIZADOR DE COMPLEJIDAD")
    
    # ========== PRUEBA 1: For Simple ==========
    print_header("PRUEBA 1: FOR SIMPLE → O(n)")
    
    code1 = load_example('for_simple.txt')
    print_analysis("Loop Simple", code1)
    
    # ========== PRUEBA 2: For Anidado ==========
    print_header("PRUEBA 2: FOR ANIDADO → O(n²)")
    
    code2 = load_example('nested_for.txt')
    print_analysis("Loops Anidados (2 niveles)", code2)
    
    # ========== PRUEBA 3: Triple For Anidado ==========
    print_header("PRUEBA 3: FOR TRIPLE ANIDADO → O(n³)")
    
    code3 = load_example('triple_for.txt')
    print_analysis("Loops Anidados (3 niveles)", code3)
    
    # ========== PRUEBA 4: While hasta n ==========
    print_header("PRUEBA 4: WHILE HASTA N → O(n)")
    
    code4 = load_example('while_n.txt')
    print_analysis("While Simple", code4)
    
    # ========== PRUEBA 5: If Simple ==========
    print_header("PRUEBA 5: IF (no cambia orden de complejidad)")
    
    code5 = load_example('for_if.txt')
    print_analysis("For con If", code5)
    
    # ========== PRUEBA 6: If con Early Return ==========
    print_header("PRUEBA 6: IF CON EARLY RETURN (afecta Ω)")
    
    code6 = load_example('linear_early_return.txt')
    print_analysis("Búsqueda Lineal con Early Return", code6)
    
    # ========== PRUEBA 7: Búsqueda Binaria ==========
    print_header("PRUEBA 7: BÚSQUEDA BINARIA → O(log n)")
    
    code7 = load_example('binary_search.txt')
    print_analysis("Búsqueda Binaria", code7)
    
    # ========== PRUEBA 8: Recursión Simple ==========
    print_header("PRUEBA 8: RECURSIÓN SIMPLE → O(n)")
    
    code8 = load_example('factorial.txt')
    print_analysis("Factorial Recursivo", code8)
    
    # ========== PRUEBA 9: Divide y Conquista (Merge Sort) ==========
    print_header("PRUEBA 9: DIVIDE Y CONQUISTA → O(n log n)")
    
    code9 = load_example('merge_sort.txt')
    print_analysis("Merge Sort (Divide y Conquista)", code9)
    
    # ========== PRUEBA 10: Fibonacci Recursivo ==========
    print_header("PRUEBA 10: FIBONACCI RECURSIVO → O(2^n)")
    
    code10 = load_example('fibonacci.txt')
    print_analysis("Fibonacci Recursivo (Exponencial)", code10)
    
    # ========== PRUEBA 11: Algoritmo Constante ==========
    print_header("PRUEBA 11: ALGORITMO CONSTANTE → O(1)")
    
    code11 = load_example('constant.txt')
    print_analysis("Operaciones Constantes", code11)
    
    # ========== PRUEBA 12: While Logarítmico ==========
    print_header("PRUEBA 12: WHILE LOGARÍTMICO → O(log n)")
    
    code12 = load_example('log_while.txt')
    print_analysis("While con división por 2", code12)

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
        print("\n1. Ejecutar todas las pruebas (recomendado)")
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
        elif choice == "3a":
            code = load_example('for_simple.txt')
            print_analysis("Loop Simple", code)
        elif choice == "3b":
            code = load_example('nested_for.txt')
            print_analysis("Loops Anidados", code)
        elif choice == "3c":
            code = load_example('binary_search.txt')
            print_analysis("Búsqueda Binaria", code)
        elif choice == "3d":
            code = load_example('factorial.txt')
            print_analysis("Factorial Recursivo", code)
        elif choice == "3e":
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