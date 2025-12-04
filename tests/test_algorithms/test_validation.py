"""
Script de prueba con debugging para validation.py
Identifica y corrige problemas en el análisis
"""

from datetime import datetime
import json
import sys
from pathlib import Path
import re
from typing import List

# Agregar raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.validation import (
    validate_efficiency, 
    EfficiencyOrchestrator,
    EfficiencyValidationResult,
    analyze_line_by_line
)
from core.complexity import ComplexityAnalyzer
# from core.patterns import PatternRecognizer
from core.parser import PseudocodeParser

# Loader local para ejemplos 
EXAMPLES_DIR = Path(__file__).parent.parent / 'pseudocode'

def _candidate_dirs():
    """Retorna directorios candidatos donde buscar archivos .txt"""
    dirs = [EXAMPLES_DIR]
    # Agregar otros directorios si existen
    alt_dirs = [
        Path(__file__).parent.parent / 'test_cases',
        Path(__file__).parent.parent / 'examples',
    ]
    dirs.extend([d for d in alt_dirs if d.exists()])
    return dirs

def load_example(name: str) -> str:
    """
    Carga un archivo de ejemplo de pseudocódigo.
    
    Args:
        name: Nombre del archivo (con o sin extensión .txt)
        
    Returns:
        Contenido del archivo
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo
    """
    import re
    
    # 1) Intentar ruta exacta
    for d in _candidate_dirs():
        p = d / name
        if p.exists():
            return p.read_text(encoding='utf-8')
        p2 = d / f"{name}.txt"
        if p2.exists():
            return p2.read_text(encoding='utf-8')
    
    # 2) Buscar por tokens en el nombre
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
        best_match = candidates[0][1]
        print(f"ℹ  Usando '{best_match.name}' como mejor coincidencia para '{name}'")
        return best_match.read_text(encoding='utf-8')
    
    # 3) Buscar por contenido
    content_candidates = []
    for d in _candidate_dirs():
        for f in d.glob('*.txt'):
            content = f.read_text(encoding='utf-8').lower()
            score = sum(1 for t in tokens if t in content)
            if score > 0:
                content_candidates.append((score, f))
    
    if content_candidates:
        content_candidates.sort(reverse=True)
        best_match = content_candidates[0][1]
        print(f"ℹ  Usando '{best_match.name}' (encontrado por contenido)")
        return best_match.read_text(encoding='utf-8')
    
    # No encontrado
    available = []
    for d in _candidate_dirs():
        available.extend([p.name for p in d.glob('*.txt')])
    
    available_str = ', '.join(sorted(available)) if available else "ninguno"
    raise FileNotFoundError(
        f" Ejemplo no encontrado: '{name}'\n"
        f"   Archivos disponibles: {available_str}\n"
        f"   Directorios buscados: {[str(d) for d in _candidate_dirs()]}"
    )

def list_available_examples() -> list:
    """Lista todos los archivos .txt disponibles"""
    examples = []
    for d in _candidate_dirs():
        if d.exists():
            examples.extend(list(d.glob('*.txt')))
    return sorted(examples, key=lambda p: p.name)

def debug_analysis(code: str, name: str = "debug") -> EfficiencyValidationResult:
    """
    Analiza paso por paso para identificar problemas en el análisis de eficiencia.
    
    Args:
        code: Pseudocódigo a analizar
        name: Nombre del algoritmo
        
    Returns:
        EfficiencyValidationResult
    """
    
    print("\n" + "="*70)
    print(f" DEBUG: Analizando '{name}'")
    print("="*70)
    
    print("\n CÓDIGO A ANALIZAR:")
    print("-"*70)
    print(code)
    print("-"*70)
    
    # ========== PASO 1: ANÁLISIS DE COMPLEJIDAD ==========
    print("\n PASO 1: Análisis de Complejidad")
    print("-"*70)
    analyzer = ComplexityAnalyzer()
    
    try:
        complexity = analyzer.analyze(code)
        print(f"✓ Big-O (Peor caso):     {complexity.big_o}")
        print(f"✓ Omega (Mejor caso):    {complexity.omega}")
        print(f"✓ Theta (Caso promedio): {complexity.theta}")
        print(f"✓ Explicación: {complexity.explanation}")
        
        # Validar coherencia básica
        if complexity.big_o and complexity.omega and complexity.theta:
            print(f"\n   Verificación de coherencia:")
            print(f"   - Notaciones presentes: ✓")
            
            # Verificar estructura del código
            loop_count = code.lower().count('for') + code.lower().count('para')
            loop_count += code.lower().count('while') + code.lower().count('mientras')
            
            if loop_count == 0 and 'O(1)' not in complexity.big_o:
                print(f"   Sin loops, pero Big-O no es O(1)")
            elif loop_count == 1 and 'O(n)' not in complexity.big_o and 'O(log' not in complexity.big_o:
                print(f"   1 loop, pero Big-O no es O(n) ni O(log n)")
            elif loop_count >= 2:
                nested = _detect_nested_structure(code)
                if nested and 'n²' not in complexity.big_o and 'n^2' not in complexity.big_o:
                    print(f"   Loops anidados, pero Big-O no es O(n²)")
                
    except Exception as e:
        print(f" Error en análisis de complejidad: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # # ========== PASO 2: ANÁLISIS DE PATRONES ==========
    # print("\n PASO 2: Análisis de Patrones Algorítmicos")
    # print("-"*70)
    # pattern_recognizer = PatternRecognizer()
    
    # try:
    #     patterns = pattern_recognizer.analyze(code)
    #     print(f"✓ Patrones detectados: {len(patterns)}")
        
    #     if patterns:
    #         for p in patterns:
    #             confidence = p.get('confidence', 'N/A')
    #             print(f"  • {p.get('name', 'unknown'):25s} | {p.get('description', '')[:40]:40s} | {p.get('complexity', 'N/A'):15s} | Confianza: {confidence}")
    #     else:
    #         print("  ℹ  No se detectaron patrones específicos")
            
    # except Exception as e:
    #     print(f" Error en análisis de patrones: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # ========== PASO 3: ANÁLISIS LÍNEA POR LÍNEA ==========
    print("\n PASO 3: Análisis de Costos por Línea")
    print("-"*70)
    parser = PseudocodeParser()
    
    try:
        lines = parser.analyze_by_line(code)
        relevant_lines = [l for l in lines if l.get('type') != 'summary']
        print(f"✓ Líneas analizadas: {len(relevant_lines)}")
        
        print(f"\n   Primeras líneas:")
        for line_info in relevant_lines[:5]:
            line_no = line_info.get('line', 0)
            line_type = line_info.get('type', 'unknown')
            time_cost = line_info.get('time_cost', 'N/A')
            exec_count = line_info.get('exec_count', 'N/A')

            time_cost_str = str(time_cost) if time_cost != 'N/A' else 'N/A'
            exec_count_str = str(exec_count) if exec_count != 'N/A' else 'N/A'
            
            print(f"   Línea {line_no:2d}: {line_type:15s} | Costo: {time_cost_str:10s} | Ejecuciones: {exec_count_str}")
        
        if len(relevant_lines) > 5:
            print(f"   ... ({len(relevant_lines) - 5} líneas más)")
            
    except Exception as e:
        print(f"❌ Error en análisis línea por línea: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== PASO 4: VALIDACIÓN CON GEMINI ==========
    print("\n PASO 4: Validación de Eficiencia con Gemini")
    print("-"*70)
    print(" Consultando Gemini... (esto puede tardar 5-10 segundos)")
    
    try:
        result = validate_efficiency(code, name, is_file=False)
        return result
        
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def _detect_nested_structure(code: str) -> bool:
    """
    Detecta si hay loops VERDADERAMENTE anidados en el código.
    CORREGIDO: Ahora detecta correctamente la anidación real.
    """
    lines = code.split('\n')
    depth = 0
    max_depth = 0
    inside_loop = False
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Ignorar líneas vacías y comentarios
        if not line_lower or line_lower.startswith('#') or line_lower.startswith('//'):
            continue
        
        # Detectar inicio de loop
        if re.search(r'\b(for|para|while|mientras)\b.*\b(hacer|do)\b', line_lower):
            depth += 1
            max_depth = max(max_depth, depth)
            inside_loop = True
        # Detectar fin de loop
        elif re.search(r'\b(fin|end)\s*(para|for|mientras|while)?\b', line_lower):
            if depth > 0:
                depth -= 1
            if depth == 0:
                inside_loop = False
    
    # Solo considerar anidación real si la profundidad máxima es >= 2
    return max_depth >= 2

def test_simple_cases():
    """Pruebas con casos simples y claros"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           PRUEBAS DE VALIDACIÓN CON DEBUGGING                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    results = []
    
    # ==================== TEST 1: Loop Simple ====================
    print("\n" + "█"*70)
    print("TEST 1: Loop Simple → Esperado: O(n)")
    print("█"*70)
    
    try:
        code1 = load_example('for_simple.txt')
        result1 = debug_analysis(code1, "loop_simple_O_n")
        
        if result1:
            print(f"\n✅ RESULTADO:")
            # print(f"   Puntuación de Eficiencia: {result1.overall_efficiency_score:.1f}/100")
            # print(f"   Rigor Matemático: {result1.mathematical_rigor}")
            
            print(f"\n   Desglose:")
            print(f"   - Complejidad: {result1.complexity_correctness:.1f}/100")
            print(f"   - Recurrencia: {result1.recurrence_correctness:.1f}/100")
            print(f"   - Costos:      {result1.cost_analysis_correctness:.1f}/100")
            
            if result1.critical_errors:
                print(f"\n     Errores críticos detectados:")
                for error in result1.critical_errors[:3]:
                    print(f"      • {error}")
            
            results.append(("Loop Simple O(n)", result1))
        else:
            results.append(("Loop Simple O(n)", None))
            
    except FileNotFoundError as e:
        print(f"\n  {e}")
        results.append(("Loop Simple O(n)", None))
    
    input("\n  Presiona Enter para continuar al siguiente test...")
    
    # ==================== TEST 2: Loops Anidados ====================
    print("\n" + "█"*70)
    print("TEST 2: Loops Anidados → Esperado: O(n²)")
    print("█"*70)
    
    try:
        code2 = load_example('nested_for.txt')
        result2 = debug_analysis(code2, "loops_anidados_O_n2")
        
        # if result2:
        #     print(f"\n RESULTADO:")
        #     print(f"   Puntuación de Eficiencia: {result2.overall_efficiency_score:.1f}/100")
        #     print(f"   Rigor Matemático: {result2.mathematical_rigor}")
            
        #     results.append(("Loops Anidados O(n²)", result2))
        # else:
        #     results.append(("Loops Anidados O(n²)", None))
            
    except FileNotFoundError as e:
        print(f"\n  {e}")
        results.append(("Loops Anidados O(n²)", None))
    
    input("\n Presiona Enter para continuar al siguiente test...")
    
    # ==================== TEST 3: Búsqueda Binaria ====================
    print("\n" + "█"*70)
    print("TEST 3: Búsqueda Binaria → Esperado: O(log n)")
    print("█"*70)
    
    try:
        code3 = load_example('binary_search.txt')
        result3 = debug_analysis(code3, "busqueda_binaria_O_log_n")
        
        # if result3:
        #     print(f"\n RESULTADO:")
        #     print(f"   Puntuación de Eficiencia: {result3.overall_efficiency_score:.1f}/100")
        #     print(f"   Rigor Matemático: {result3.mathematical_rigor}")
            
        #     results.append(("Búsqueda Binaria O(log n)", result3))
        # else:
        #     results.append(("Búsqueda Binaria O(log n)", None))
            
    except FileNotFoundError as e:
        print(f"\n  {e}")
        results.append(("Búsqueda Binaria O(log n)", None))
    
    # ==================== TEST 4: Constante ====================
    print("\n" + "█"*70)
    print("TEST 4: Sin Loops → Esperado: O(1)")
    print("█"*70)
    
    try:
        code4 = load_example('constant.txt')
        result4 = debug_analysis(code4, "constante_O_1")
        
        # if result4:
        #     print(f"\n RESULTADO:")
        #     print(f"   Puntuación de Eficiencia: {result4.overall_efficiency_score:.1f}/100")
        #     print(f"   Rigor Matemático: {result4.mathematical_rigor}")
            
        #     results.append(("Constante O(1)", result4))
        # else:
        #     results.append(("Constante O(1)", None))
            
    except FileNotFoundError as e:
        print(f"\n  {e}")
        results.append(("Constante O(1)", None))
    
    # ==================== RESUMEN FINAL ====================
    print("\n" + "="*70)
    print(" RESUMEN DE PRUEBAS")
    print("="*70)
    print(f"{'Algoritmo':<35s} {'Puntuación':>12s} {'Rigor':>10s} {'Estado':>8s}")
    print("-"*70)
    
    for name, result in results:
        if result:
            score = result.overall_efficiency_score
            rigor = result.mathematical_rigor
            
            if score >= 85:
                status = " PASS"
            elif score >= 60:
                status = "  WARN"
            else:
                status = " FAIL"
            
            print(f"{name:<35s} {score:>10.1f}/100 {rigor:>10s} {status:>8s}")
        else:
            print(f"{name:<35s} {'N/A':>12s} {'N/A':>10s} {' ERROR':>8s}")
    
    print("="*70)
    
    # Estadísticas
    valid_results = [r for _, r in results if r is not None]
    if valid_results:
        avg_score = sum(r.overall_efficiency_score for r in valid_results) / len(valid_results)
        print(f"\n Puntuación promedio: {avg_score:.1f}/100")
        
        high_rigor = sum(1 for r in valid_results if r.mathematical_rigor == 'HIGH')
        print(f"   Algoritmos con ALTO rigor: {high_rigor}/{len(valid_results)}")


def quick_fix_test() -> bool:
    """
    Prueba rápida para verificar que el análisis de eficiencia funciona correctamente.
    
    Returns:
        True si todo está correcto, False si hay errores
    """
    
    print("\n" + "="*70)
    print(" PRUEBA RÁPIDA DE VERIFICACIÓN")
    print("="*70)
    
    # Código super simple: 1 loop
    try:
        code = load_example('for_simple.txt')
    except FileNotFoundError:
        # Fallback a código inline
        code = """
PARA i DESDE 0 HASTA n-1 HACER
    ESCRIBIR i
FIN PARA
"""
    
    print("\n Código:")
    print("-"*70)
    print(code)
    print("-"*70)
    
    # 1. Verificar análisis automático
    print("\n  Verificando análisis automático de complejidad...")
    analyzer = ComplexityAnalyzer()
    complexity = analyzer.analyze(code)
    
    print(f"   Big-O detectado: {complexity.big_o}")
    print(f"   Explicación: {complexity.explanation}")
    
    expected = "O(n)"
    if complexity.big_o == expected:
        print(f"    Correcto! (esperado: {expected})")
    else:
        print(f"    INCORRECTO! (esperado: {expected}, obtenido: {complexity.big_o})")
        print("\n    PROBLEMA DETECTADO:")
        print("      El analizador de complejidad está dando resultados incorrectos.")
        print("      Posibles causas:")
        print("      - Parser.analyze_by_line() está contando mal los loops")
        print("      - ComplexityAnalyzer._analyze_loops() tiene un bug")
        return False
    
    # 2. Validar con sistema completo
    print("\n Validando con sistema de eficiencia completo...")
    result = validate_efficiency(code, "quick_test", is_file=False)
    
    # print(f"\n   Puntuación de Eficiencia: {result.overall_efficiency_score:.1f}/100")
    # print(f"   Rigor Matemático: {result.mathematical_rigor}")
    
    if result.overall_efficiency_score >= 70:
        print("\n TODO FUNCIONA CORRECTAMENTE!")
        return True
    else:
        print("\n  Baja puntuación - revisar análisis")
        
        if result.critical_errors:
            print("\n   Errores críticos:")
            for error in result.critical_errors:
                print(f"      • {error}")
        
        return False

def interactive_debug():
    """Debug interactivo para pseudocódigo personalizado"""
    
    print("\n" + "="*70)
    print(" MODO DEBUG INTERACTIVO")
    print("="*70)
    print("\n Ingresa tu pseudocódigo (línea vacía para terminar):")
    print("   Ejemplo:")
    print("   PARA i DESDE 0 HASTA n-1 HACER")
    print("       ESCRIBIR i")
    print("   FIN PARA")
    print()
    
    lines = []
    while True:
        try:
            line = input()
            if not line.strip() and lines:
                break
            if line.strip():
                lines.append(line)
        except EOFError:
            break
    
    code = "\n".join(lines)
    
    if not code.strip():
        print(" No se ingresó código")
        return
    
    name = input("\n  Nombre del algoritmo (opcional): ").strip() or "debug_manual"
    
    result = debug_analysis(code, name)
    
    if result:
        # Exportar reporte
        orchestrator = EfficiencyOrchestrator()
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / f"{name}_debug.json"
        orchestrator.export_validation_report(result, str(filepath))
        
        print(f"\n Reporte exportado: {filepath}")
        
        # Mostrar resumen
        print(f"\n RESUMEN:")
        print(f"   Puntuación: {result.overall_efficiency_score:.1f}/100")
        print(f"   Rigor: {result.mathematical_rigor}")
        
        if result.corrections:
            print(f"\n Sugerencias de corrección:")
            for correction in result.corrections:
                print(f"   • {correction}")

def test_directory():
    """Valida todos los archivos en el directorio de ejemplos"""
    
    print("\n" + "="*70)
    print(" VALIDACIÓN DE DIRECTORIO COMPLETO")
    print("="*70)
    
    if not EXAMPLES_DIR.exists():
        print(f" Directorio no existe: {EXAMPLES_DIR}")
        return
    
    examples = list(EXAMPLES_DIR.glob("*.txt"))
    
    if not examples:
        print(f"  No se encontraron archivos .txt en {EXAMPLES_DIR}")
        return
    
    print(f"\n Directorio: {EXAMPLES_DIR}")
    print(f" Archivos encontrados: {len(examples)}")
    print()
    
    for idx, example in enumerate(examples, 1):
        size = example.stat().st_size
        print(f"   {idx:2d}. {example.name:40s} ({size} bytes)")
    
    print(f"\n{'─'*70}")
    print("Opciones:")
    print("  • Número(s) específico(s): Ej. '1' o '1,3,5' o '1-3'")
    print("  • 'all' o 'a' para validar todos")
    print("  • Enter para cancelar")
    print("─"*70)
    
    choice = input("\nSelección: ").strip().lower()
    
    if not choice:
        print("Cancelado")
        return
    
    # Determinar archivos a procesar
    selected_files = []
    
    if choice in ['all', 'a']:
        selected_files = examples
        print(f"\n✓ Validando TODOS los archivos ({len(examples)})")
    else:
        # Parsear selección
        try:
            indices = set()
            
            for part in choice.split(','):
                part = part.strip()
                
                if '-' in part:
                    # Rango: 1-3
                    start, end = map(int, part.split('-'))
                    indices.update(range(start, end + 1))
                else:
                    # Número único: 5
                    indices.add(int(part))
            
            # Validar índices
            valid_indices = [i for i in indices if 1 <= i <= len(examples)]
            
            if not valid_indices:
                print("No se seleccionaron archivos válidos")
                return
            
            selected_files = [examples[i-1] for i in sorted(valid_indices)]
            print(f"\n✓ Validando {len(selected_files)} archivo(s):")
            for f in selected_files:
                print(f"   • {f.name}")
            
        except ValueError:
            print("Formato inválido. Usa números, rangos (1-3) o 'all'")
            return
    
    confirm = input(f"\n¿Continuar? (s/N): ").strip().lower()
    
    if confirm != 's':
        print("Cancelado")
        return
    
    # Validar archivos seleccionados
    output_dir = Path("validation_reports")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO {len(selected_files)} ARCHIVO(S)")
    print(f"{'='*70}\n")
    
    for idx, txt_file in enumerate(selected_files, 1):
        print(f"\n[{idx}/{len(selected_files)}] Procesando: {txt_file.name}")
        print("─"*70)
        
        try:
            result = validate_efficiency(str(txt_file), is_file=True)
            results.append(result)
            
            # Mostrar resumen rápido
            print(f"✓ Completado: {result.overall_efficiency_score:.1f}/100 ({result.mathematical_rigor})")
            
            # Exportar reporte individual
            report_name = f"{txt_file.stem}_validation.json"
            report_path = output_dir / report_name
            
            orchestrator = EfficiencyOrchestrator()
            orchestrator.export_validation_report(result, str(report_path))
            
        except Exception as e:
            print(f" Error procesando {txt_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generar resumen consolidado
    if results:
        _generate_quick_summary(results, output_dir, selected_files)

def _generate_quick_summary(
    results: List[EfficiencyValidationResult],
    output_dir: Path,
    files: List[Path]
):
    """Genera resumen consolidado rápido"""
    
    print(f"\n{'='*70}")
    print(" RESUMEN CONSOLIDADO")
    print(f"{'='*70}")
    
    # Estadísticas
    scores = [r.overall_efficiency_score for r in results]
    avg_score = sum(scores) / len(scores)
    
    print(f"\n Estadísticas Generales:")
    print(f"   Total archivos: {len(results)}")
    print(f"   Puntuación promedio: {avg_score:.1f}/100")
    print(f"   Rango: {min(scores):.1f} - {max(scores):.1f}")
    
    # Distribución por rigor
    high = sum(1 for r in results if r.mathematical_rigor == 'HIGH')
    medium = sum(1 for r in results if r.mathematical_rigor == 'MEDIUM')
    low = sum(1 for r in results if r.mathematical_rigor == 'LOW')
    
    print(f"\n Distribución por Rigor Matemático:")
    print(f"   HIGH:   {high:2d} ({high/len(results)*100:.0f}%)")
    print(f"   MEDIUM: {medium:2d} ({medium/len(results)*100:.0f}%)")
    print(f"   LOW:    {low:2d} ({low/len(results)*100:.0f}%)")
    
    # Tabla de resultados
    print(f"\n Resultados Detallados:")
    print("─"*70)
    print(f"{'Archivo':<30s} {'Score':>10s} {'Rigor':>10s} {'Estado':>10s}")
    print("─"*70)
    
    for result in results:
        name = result.algorithm_name[:28]
        score = result.overall_efficiency_score
        rigor = result.mathematical_rigor
        
        if score >= 85:
            status = "PASS"
        elif score >= 60:
            status = "WARN"
        else:
            status = "FAIL"
        
        print(f"{name:<30s} {score:>8.1f}/100 {rigor:>10s} {status:>10s}")
    
    print("─"*70)
    
    # Guardar resumen JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(results),
        'statistics': {
            'avg_score': avg_score,
            'min_score': min(scores),
            'max_score': max(scores),
            'high_rigor': high,
            'medium_rigor': medium,
            'low_rigor': low
        },
        'results': [
            {
                'algorithm': r.algorithm_name,
                'file': r.file_path,
                'score': r.overall_efficiency_score,
                'rigor': r.mathematical_rigor,
                'critical_errors': len(r.critical_errors)
            }
            for r in results
        ]
    }
    
    summary_path = output_dir / "batch_summary.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n Reportes guardados en: {output_dir}")
    print(f"   • Resumen: {summary_path.name}")
    print(f"   • Individuales: {len(results)} archivos JSON")
    print(f"{'='*70}\n")

def show_menu():
    """Muestra el menú principal de opciones"""
    print("\n" + "=" * 70)
    print(" SISTEMA DE PRUEBAS - VALIDACIÓN DE EFICIENCIA ALGORÍTMICA")
    print("=" * 70)
    print("\nOpciones disponibles:")
    print()
    print("  1. Prueba rápida (verificar corrección básica)")
    print("  2. Suite completa (casos predefinidos con debugging)")
    print("  3. Debug interactivo (tu propio código)")
    print("  4. Validar directorio completo")
    print("  5. Listar archivos disponibles")
    print("  6. Análisis línea por línea y ecuación de recurrencia")  # Nueva opción
    print("  0. Salir")
    print()

def main():
    """Función principal con menú interactivo"""
    while True:
        show_menu()
        try:
            choice = input("Selecciona una opción (0-6): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n ¡Hasta luego!")
            sys.exit(0)

        print()

        if choice == "1":
            success = quick_fix_test()
            if not success:
                print("\n  Se detectaron problemas en el análisis")

        elif choice == "2":
            test_simple_cases()

        elif choice == "3":
            interactive_debug()

        elif choice == "4":
            test_directory()

        elif choice == "5":
            examples = list_available_examples()
            print(" Archivos disponibles:")
            print("-" * 70)
            for ex in examples:
                print(f"   • {ex.name:40s} ({ex.stat().st_size} bytes)")
            print("-" * 70)
            print(f"Total: {len(examples)} archivos")

        elif choice == "6":  # Nueva opción
            analyze_line_by_line()

        elif choice == "0":
            print(" ¡Hasta luego!")
            sys.exit(0)

        else:
            print(" Opción inválida. Intenta de nuevo.")

        input("\n  Presiona Enter para continuar...")

if __name__ == "__main__":
    # Si se pasa un argumento, usarlo como modo directo
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "quick":
            success = quick_fix_test()
            sys.exit(0 if success else 1)
            
        elif mode == "suite":
            test_simple_cases()
            sys.exit(0)
            
        elif mode == "dir":
            test_directory()
            sys.exit(0)
            
        else:
            print(f" Modo desconocido: {mode}")
            print("Modos disponibles: quick, suite, dir")
            sys.exit(1)
    
    # Modo interactivo
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrumpido por el usuario. ¡Hasta luego!")
        sys.exit(0)