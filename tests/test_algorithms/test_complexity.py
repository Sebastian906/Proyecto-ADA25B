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
    
    code1 = """
    for i = 0 to n do
        print i
    end
    """
    print_analysis("Loop Simple", code1)
    
    # ========== PRUEBA 2: For Anidado ==========
    print_header("PRUEBA 2: FOR ANIDADO → O(n²)")
    
    code2 = """
    for i = 0 to n do
        for j = 0 to n do
            print i + j
        end
    end
    """
    print_analysis("Loops Anidados (2 niveles)", code2)
    
    # ========== PRUEBA 3: Triple For Anidado ==========
    print_header("PRUEBA 3: FOR TRIPLE ANIDADO → O(n³)")
    
    code3 = """
    for i = 0 to n do
        for j = 0 to n do
            for k = 0 to n do
                print i + j + k
            end
        end
    end
    """
    print_analysis("Loops Anidados (3 niveles)", code3)
    
    # ========== PRUEBA 4: While hasta n ==========
    print_header("PRUEBA 4: WHILE HASTA N → O(n)")
    
    code4 = """
    i = 0
    while i < n do
        print i
        i = i + 1
    end
    """
    print_analysis("While Simple", code4)
    
    # ========== PRUEBA 5: If Simple ==========
    print_header("PRUEBA 5: IF (no cambia orden de complejidad)")
    
    code5 = """
    for i = 0 to n do
        if i > 5 then
            print i
        end
    end
    """
    print_analysis("For con If", code5)
    
    # ========== PRUEBA 6: If con Early Return ==========
    print_header("PRUEBA 6: IF CON EARLY RETURN (afecta Ω)")
    
    code6 = """
    for i = 0 to n do
        if arr[i] == target then
            return i
        end
    end
    return -1
    """
    print_analysis("Búsqueda Lineal con Early Return", code6)
    
    # ========== PRUEBA 7: Búsqueda Binaria ==========
    print_header("PRUEBA 7: BÚSQUEDA BINARIA → O(log n)")
    
    code7 = """
    function busquedaBinaria(arr, x)
        izquierda = 0
        derecha = n - 1
        while izquierda <= derecha do
            medio = (izquierda + derecha) / 2
            if arr[medio] == x then
                return medio
            end
            if arr[medio] < x then
                izquierda = medio + 1
            else
                derecha = medio - 1
            end
        end
        return -1
    end
    """
    print_analysis("Búsqueda Binaria", code7)
    
    # ========== PRUEBA 8: Recursión Simple ==========
    print_header("PRUEBA 8: RECURSIÓN SIMPLE → O(n)")
    
    code8 = """
    function factorial(n)
        if n <= 1 then
            return 1
        end
        return n * factorial(n - 1)
    end
    """
    print_analysis("Factorial Recursivo", code8)
    
    # ========== PRUEBA 9: Divide y Conquista (Merge Sort) ==========
    print_header("PRUEBA 9: DIVIDE Y CONQUISTA → O(n log n)")
    
    code9 = """
    function mergeSort(arr, inicio, fin)
        if inicio < fin then
            medio = (inicio + fin) / 2
            mergeSort(arr, inicio, medio)
            mergeSort(arr, medio + 1, fin)
            merge(arr, inicio, medio, fin)
        end
    end
    """
    print_analysis("Merge Sort (Divide y Conquista)", code9)
    
    # ========== PRUEBA 10: Fibonacci Recursivo ==========
    print_header("PRUEBA 10: FIBONACCI RECURSIVO → O(2^n)")
    
    code10 = """
    function fibonacci(n)
        if n <= 1 then
            return n
        end
        return fibonacci(n-1) + fibonacci(n-2)
    end
    """
    print_analysis("Fibonacci Recursivo (Exponencial)", code10)
    
    # ========== PRUEBA 11: Algoritmo Constante ==========
    print_header("PRUEBA 11: ALGORITMO CONSTANTE → O(1)")
    
    code11 = """
    x = 5
    y = 10
    z = x + y
    print z
    """
    print_analysis("Operaciones Constantes", code11)
    
    # ========== PRUEBA 12: While Logarítmico ==========
    print_header("PRUEBA 12: WHILE LOGARÍTMICO → O(log n)")
    
    code12 = """
    i = n
    while i > 1 do
        i = i / 2
        print i
    end
    """
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
            code = "for i = 0 to n do\n    print i\nend"
            print_analysis("Loop Simple", code)
        elif choice == "3b":
            code = "for i = 0 to n do\n    for j = 0 to n do\n        print i\n    end\nend"
            print_analysis("Loops Anidados", code)
        elif choice == "3c":
            code = """izquierda = 0
derecha = n-1
while izquierda <= derecha do
    medio = (izquierda + derecha) / 2
end"""
            print_analysis("Búsqueda Binaria", code)
        elif choice == "3d":
            code = """function factorial(n)
    if n <= 1 then
        return 1
    end
    return n * factorial(n-1)
end"""
            print_analysis("Factorial Recursivo", code)
        elif choice == "3e":
            code = """function mergeSort(arr, l, r)
    if l < r then
        m = (l + r) / 2
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)
    end
end"""
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