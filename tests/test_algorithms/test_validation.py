"""
Script de prueba con debugging para validation.py
Identifica y corrige problemas en el análisis
"""

import sys
from pathlib import Path

# Agregar raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.validation import validate_algorithm, ValidationOrchestrator
from core.complexity import ComplexityAnalyzer
from core.patterns import PatternRecognizer
from core.parser import PseudocodeParser
from pathlib import Path

# Loader local para ejemplos (usar la carpeta oficial `pseudocode`)
EXAMPLES_DIR = Path(__file__).parent.parent / 'pseudocode'

def _candidate_dirs():
    # Solo buscar en la carpeta canonical `pseudocode`
    dirs = [EXAMPLES_DIR]
    return dirs


def load_example(name: str) -> str:
    import re
    # 1) Intentar ruta exacta en dirs candidatas
    for d in _candidate_dirs():
        p = d / name
        if p.exists():
            return p.read_text(encoding='utf-8')
        p2 = d / f"{name}.txt"
        if p2.exists():
            return p2.read_text(encoding='utf-8')

    # 2) Buscar por tokens en el nombre del archivo
    key = Path(name).stem.lower()
    tokens = re.findall(r"\w+", key)
    candidates = []
    for d in _candidate_dirs():
        for f in d.glob('*.txt'):
            fname = f.name.lower()
            score = sum(1 for t in tokens if t in fname)
            if score > 0:
                candidates.append((score, f))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1].read_text(encoding='utf-8')

    # 3) Buscar por contenido (tokens dentro del contenido)
    content_candidates = []
    for d in _candidate_dirs():
        for f in d.glob('*.txt'):
            content = f.read_text(encoding='utf-8').lower()
            score = sum(1 for t in tokens if t in content)
            if score > 0:
                content_candidates.append((score, f))
    if content_candidates:
        content_candidates.sort(reverse=True)
        return content_candidates[0][1].read_text(encoding='utf-8')

    available = ', '.join([p.name for d in _candidate_dirs() for p in d.glob('*.txt')])
    raise FileNotFoundError(f"Ejemplo no encontrado: {name}. Archivos disponibles: {available}")


def debug_analysis(code: str, name: str = "debug"):
    """Analiza paso por paso para encontrar el problema"""
    
    print("\n" + "="*70)
    print(f"DEBUG: Analizando '{name}'")
    print("="*70)
    
    print("\n CÓDIGO A ANALIZAR:")
    print("-"*70)
    print(code)
    print("-"*70)
    
    # 1. Análisis de Complejidad
    print("\n PASO 1: Análisis de Complejidad")
    print("-"*70)
    analyzer = ComplexityAnalyzer()
    try:
        complexity = analyzer.analyze(code)
        print(f"✓ Big-O: {complexity.big_o}")
        print(f"✓ Omega: {complexity.omega}")
        print(f"✓ Theta: {complexity.theta}")
        print(f"✓ Explicación: {complexity.explanation}")
    except Exception as e:
        print(f"✗ Error en análisis de complejidad: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Análisis de Patrones
    print("\n PASO 2: Análisis de Patrones")
    print("-"*70)
    pattern_recognizer = PatternRecognizer()
    try:
        patterns = pattern_recognizer.analyze(code)
        print(f"✓ Patrones detectados: {len(patterns)}")
        for p in patterns:
            print(f"  • {p.get('name')}: {p.get('description')} [{p.get('complexity')}]")
    except Exception as e:
        print(f"✗ Error en análisis de patrones: {e}")
    
    # 3. Análisis Línea por Línea
    print("\n PASO 3: Análisis Línea por Línea")
    print("-"*70)
    parser = PseudocodeParser()
    try:
        lines = parser.analyze_by_line(code)
        print(f"✓ Líneas analizadas: {len(lines)}")
        for line_info in lines[:5]:  # Mostrar primeras 5
            if line_info.get('type') != 'summary':
                print(f"  Línea {line_info.get('line')}: {line_info.get('type')} - {line_info.get('time_cost')}")
    except Exception as e:
        print(f"✗ Error en análisis de líneas: {e}")
    
    # 4. Validación completa
    print("\n PASO 4: Validación con Gemini")
    print("-"*70)
    print("Consultando Gemini... (esto puede tardar 5-10 segundos)")
    
    try:
        result = validate_algorithm(code, name)
        return result
    except Exception as e:
        print(f"✗ Error en validación: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_cases():
    """Pruebas con casos simples y claros"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           PRUEBAS DE VALIDACIÓN CON DEBUGGING                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ==================== TEST 1: Loop Simple ====================
    print("\n" + "█"*70)
    print("TEST 1: Loop Simple O(n)")
    print("█"*70)
    
    code1 = load_example('for_simple.txt')
    
    result1 = debug_analysis(code1, "loop_simple_O_n")
    
    if result1:
        print(f"\n RESULTADO: Confianza = {result1.overall_confidence:.2f}%")
        if result1.overall_confidence < 50:
            print("  PROBLEMA: Baja confianza detectada")
            print("\nAnálisis Automático:")
            print(f"  Big-O: {result1.auto_analysis['complexity']['big_o']}")
            print("\nDiferencias:")
            for diff in result1.complexity_differences:
                print(f"  • {diff}")
    
    input("\n⏸  Presiona Enter para continuar al siguiente test...")
    
    # ==================== TEST 2: Loops Anidados ====================
    print("\n" + "█"*70)
    print("TEST 2: Loops Anidados O(n²)")
    print("█"*70)
    
    code2 = load_example('nested_for.txt')
    
    result2 = debug_analysis(code2, "loops_anidados_O_n2")
    
    if result2:
        print(f"\n RESULTADO: Confianza = {result2.overall_confidence:.2f}%")
    
    input("\n⏸ Presiona Enter para continuar al siguiente test...")
    
    # ==================== TEST 3: Búsqueda Binaria ====================
    print("\n" + "█"*70)
    print("TEST 3: Búsqueda Binaria O(log n)")
    print("█"*70)
    
    code3 = load_example('binary_search.txt')
    
    result3 = debug_analysis(code3, "busqueda_binaria_O_log_n")
    
    if result3:
        print(f"\n RESULTADO: Confianza = {result3.overall_confidence:.2f}%")
    
    # ==================== RESUMEN FINAL ====================
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    
    results = [
        ("Loop Simple O(n)", result1),
        ("Loops Anidados O(n²)", result2),
        ("Búsqueda Binaria O(log n)", result3)
    ]
    
    for name, result in results:
        if result:
            confidence = result.overall_confidence
            status = "✓" if confidence >= 50 else "✗"
            print(f"{status} {name:30s} Confianza: {confidence:5.1f}%")
        else:
            print(f"✗ {name:30s} ERROR")
    
    print("="*70)


def quick_fix_test():
    """Prueba rápida para verificar el fix"""
    
    print("\n" + "="*70)
    print("PRUEBA RÁPIDA DE VERIFICACIÓN")
    print("="*70)
    
    # Código super simple
    code = load_example('for_simple.txt')
    
    print("\nCódigo:")
    print(code)
    
    # Análisis manual primero
    print("\n1. Verificando análisis automático...")
    analyzer = ComplexityAnalyzer()
    complexity = analyzer.analyze(code)
    
    print(f"   Big-O detectado: {complexity.big_o}")
    print(f"   Explicación: {complexity.explanation}")
    
    # Verificar si es correcto
    expected = "O(n)"
    if complexity.big_o == expected:
        print(f"   ✓ Correcto! (esperado: {expected})")
    else:
        print(f"   ✗ INCORRECTO! (esperado: {expected}, obtenido: {complexity.big_o})")
        print("\n  PROBLEMA DETECTADO:")
        print("   El analizador de complejidad está dando resultados incorrectos.")
        print("   Posibles causas:")
        print("   - Parser.analyze_by_line() está contando mal los loops")
        print("   - ComplexityAnalyzer._analyze_loops() tiene un bug")
        return False
    
    # Ahora validar con Gemini
    print("\n2. Validando con Gemini...")
    result = validate_algorithm(code, "quick_fix_test")
    
    print(f"\n✓ Confianza: {result.overall_confidence:.2f}%")
    
    if result.overall_confidence >= 70:
        print("✓ TODO FUNCIONA CORRECTAMENTE!")
        return True
    else:
        print(" Baja confianza - revisar análisis")
        return False


def interactive_debug():
    """Debug interactivo para tu propio código"""
    
    print("\n" + "="*70)
    print("MODO DEBUG INTERACTIVO")
    print("="*70)
    print("\nIngresa tu pseudocódigo (línea vacía para terminar):")
    
    lines = []
    while True:
        line = input()
        if not line.strip() and lines:
            break
        if line.strip():
            lines.append(line)
    
    code = "\n".join(lines)
    
    if not code.strip():
        print("✗ No se ingresó código")
        return
    
    name = input("\nNombre del algoritmo: ").strip() or "debug_manual"
    
    result = debug_analysis(code, name)
    
    if result:
        # Exportar reporte
        orchestrator = ValidationOrchestrator()
        Path("reports").mkdir(exist_ok=True)
        filepath = f"reports/{name}_debug.json"
        orchestrator.export_validation_report(result, filepath)
        print(f"\n✓ Reporte exportado: {filepath}")


if __name__ == "__main__":
    import sys
    
    print("Selecciona una opción:")
    print("1. Prueba rápida (verificar si hay bug)")
    print("2. Suite completa con debugging")
    print("3. Debug interactivo (tu código)")
    
    choice = input("\nOpción (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        success = quick_fix_test()
        sys.exit(0 if success else 1)
    elif choice == "2":
        test_simple_cases()
    elif choice == "3":
        interactive_debug()
    else:
        print("Opción inválida")
        sys.exit(1)