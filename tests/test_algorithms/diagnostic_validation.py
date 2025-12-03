"""
Herramienta de diagnóstico para identificar por qué las puntuaciones son bajas.
Compara lado a lado: Automático vs Gemini
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.complexity import ComplexityAnalyzer
from core.parser import PseudocodeParser
# from core.patterns import PatternRecognizer
from llm.integration import ask_gemini
import re


def diagnose_algorithm(code: str, algorithm_name: str = "test"):
    """
    Diagnóstico detallado de un algoritmo para identificar discrepancias.
    """
    print("\n" + "="*80)
    print(f" DIAGNÓSTICO DETALLADO: {algorithm_name}")
    print("="*80)
    
    print("\n CÓDIGO:")
    print("-"*80)
    print(code)
    print("-"*80)
    
    # ========== ANÁLISIS AUTOMÁTICO ==========
    print("\n" + "="*80)
    print("  ANÁLISIS AUTOMÁTICO (Nuestro Sistema)")
    print("="*80)
    
    analyzer = ComplexityAnalyzer()
    complexity = analyzer.analyze(code)
    
    print(f"\n Complejidades:")
    print(f"   Big-O:  {complexity.big_o}")
    print(f"   Omega:  {complexity.omega}")
    print(f"   Theta:  {complexity.theta}")
    
    parser = PseudocodeParser()
    lines = parser.analyze_by_line(code)
    
    # Extraer recurrencia
    # recurrence = None
    # for item in lines:
    #     if item.get('type') == 'summary' and 'recurrence' in item:
    #         recurrence = item.get('recurrence')
    #         break
    
    # if recurrence:
    #     print(f"\n Recurrencia:")
    #     print(f"   {recurrence}")
    
    # Patrones
    # recognizer = PatternRecognizer()
    # patterns = recognizer.analyze(code)
    
    # if patterns:
    #     print(f"\n Patrones detectados:")
    #     for p in patterns:
    #         print(f"   • {p.get('name')}: {p.get('confidence')}")
    
    # ========== ANÁLISIS DE GEMINI ==========
    print("\n" + "="*80)
    print("  ANÁLISIS DE GEMINI (Referencia)")
    print("="*80)
    
    prompt = f"""Analiza este pseudocódigo y proporciona EXACTAMENTE:

1. Peor caso (Big-O): [usa formato O(...)]
2. Mejor caso (Omega): [usa formato Ω(...) o Omega(...)]
3. Caso promedio (Theta): [usa formato Θ(...) o Theta(...)]

CÓDIGO:
```
{code}
```

IMPORTANTE: Usa notaciones estándar como O(n), O(n²), O(log n), etc.
Responde de forma concisa y estructurada."""
    
    print("\n Consultando Gemini...")
    gemini_response = ask_gemini(prompt)
    
    print("\n Respuesta de Gemini:")
    print("-"*80)
    print(gemini_response)
    print("-"*80)
    
    # ========== EXTRACCIÓN Y COMPARACIÓN ==========
    print("\n" + "="*80)
    print(" COMPARACIÓN Y ANÁLISIS DE DISCREPANCIAS")
    print("="*80)
    
    # Extraer complejidades de Gemini
    gemini_complexities = extract_complexities_from_gemini(gemini_response)
    
    print("\n Comparación de Complejidades:")
    print("-"*80)
    print(f"{'Notación':<15s} {'Automático':<20s} {'Gemini':<20s} {'Coincide?':<10s}")
    print("-"*80)
    
    # Big-O
    auto_big_o = normalize_complexity(complexity.big_o)
    gemini_big_o = normalize_complexity(gemini_complexities.get('big_o', 'No detectado'))
    match_big_o = auto_big_o == gemini_big_o or are_equivalent(auto_big_o, gemini_big_o)
    print(f"{'Big-O':<15s} {complexity.big_o:<20s} {gemini_complexities.get('big_o', 'N/A'):<20s} {'✅' if match_big_o else '❌':<10s}")
    
    # Omega
    auto_omega = normalize_complexity(complexity.omega)
    gemini_omega = normalize_complexity(gemini_complexities.get('omega', 'No detectado'))
    match_omega = auto_omega == gemini_omega or are_equivalent(auto_omega, gemini_omega)
    print(f"{'Omega':<15s} {complexity.omega:<20s} {gemini_complexities.get('omega', 'N/A'):<20s} {'✅' if match_omega else '❌':<10s}")
    
    # Theta
    auto_theta = normalize_complexity(complexity.theta)
    gemini_theta = normalize_complexity(gemini_complexities.get('theta', 'No detectado'))
    match_theta = auto_theta == gemini_theta or are_equivalent(auto_theta, gemini_theta)
    print(f"{'Theta':<15s} {complexity.theta:<20s} {gemini_complexities.get('theta', 'N/A'):<20s} {'✅' if match_theta else '❌':<10s}")
    
    print("-"*80)
    
    # Calcular puntuación manual
    matches = sum([match_big_o, match_omega, match_theta])
    manual_score = (matches / 3) * 100
    
    print(f"\n Puntuación Manual de Complejidad: {manual_score:.1f}%")
    print(f"   Coincidencias: {matches}/3")
    
    # ========== ANÁLISIS DE DISCREPANCIAS ==========
    print("\n" + "="*80)
    print(" ANÁLISIS DE CAUSAS DE DISCREPANCIAS")
    print("="*80)
    
    issues = []
    
    if not match_big_o:
        issues.append({
            'type': 'Big-O',
            'auto': complexity.big_o,
            'gemini': gemini_complexities.get('big_o', 'N/A'),
            'reason': analyze_discrepancy(complexity.big_o, gemini_complexities.get('big_o'), code)
        })
    
    if not match_omega:
        issues.append({
            'type': 'Omega',
            'auto': complexity.omega,
            'gemini': gemini_complexities.get('omega', 'N/A'),
            'reason': analyze_discrepancy(complexity.omega, gemini_complexities.get('omega'), code)
        })
    
    if not match_theta:
        issues.append({
            'type': 'Theta',
            'auto': complexity.theta,
            'gemini': gemini_complexities.get('theta', 'N/A'),
            'reason': analyze_discrepancy(complexity.theta, gemini_complexities.get('theta'), code)
        })
    
    if issues:
        print("\n PROBLEMAS DETECTADOS:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue['type']}:")
            print(f"   Automático: {issue['auto']}")
            print(f"   Gemini:     {issue['gemini']}")
            print(f"   Causa:      {issue['reason']}")
            print()
    else:
        print("\n NO HAY DISCREPANCIAS - ANÁLISIS PERFECTO!")
    
    # ========== RECOMENDACIONES ==========
    print("\n" + "="*80)
    print(" RECOMENDACIONES PARA MEJORAR PUNTUACIÓN")
    print("="*80)
    
    recommendations = generate_recommendations(issues, complexity, gemini_complexities, code)
    
    if recommendations:
        for rec in recommendations:
            print(f"• {rec}")
    else:
        print(" El análisis es correcto. No hay recomendaciones.")
    
    print("\n" + "="*80)


def extract_complexities_from_gemini(text: str) -> dict:
    """Extrae complejidades del texto de Gemini con múltiples estrategias."""
    complexities = {
        'big_o': None,
        'omega': None,
        'theta': None
    }
    
    text_lower = text.lower()
    
    # Buscar las complejidades en el texto
    big_o_match = re.search(r'peor caso \(big-o\):\s*o\(([^)]+)\)', text, re.IGNORECASE)
    omega_match = re.search(r'mejor caso \(omega\):\s*ω\(([^)]+)\)', text, re.IGNORECASE)
    theta_match = re.search(r'caso promedio \(theta\):\s*θ\(([^)]+)\)', text, re.IGNORECASE)
    
    # Asignar los valores encontrados a las claves esperadas
    if big_o_match:
        complexities['big_o'] = f"O({big_o_match.group(1).strip()})"
    if omega_match:
        complexities['omega'] = f"Ω({omega_match.group(1).strip()})"
    if theta_match:
        complexities['theta'] = f"Θ({theta_match.group(1).strip()})"
    
    # Depuración: Imprimir las complejidades extraídas
    print("Complejidades extraídas de Gemini:", complexities)
    
    # Estrategia 1: Buscar patrones explícitos
    patterns = {
        'big_o': [
            r'peor\s+caso[:\s]+[o|O]\s*\(([^)]+)\)',
            r'big-o[:\s]+[o|O]\s*\(([^)]+)\)',
            r'o\(([^)]+)\)\s*(?:peor|worst)',
        ],
        'omega': [
            r'mejor\s+caso[:\s]+[ω|Ω|omega|OMEGA]\s*\(([^)]+)\)',
            r'omega[:\s]+[ω|Ω]\s*\(([^)]+)\)',
            r'[ω|Ω]\(([^)]+)\)\s*(?:mejor|best)',
        ],
        'theta': [
            r'caso\s+promedio[:\s]+[θ|Θ|theta|THETA]\s*\(([^)]+)\)',
            r'theta[:\s]+[θ|Θ]\s*\(([^)]+)\)',
            r'[θ|Θ]\(([^)]+)\)\s*(?:promedio|average)',
        ]
    }
    
    for complexity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text_lower)
            if match:
                complexities[complexity_type] = f"O({match.group(1).strip()})"
                break
    
    # Estrategia 2: Buscar cualquier notación O(...), Ω(...), Θ(...)
    if not complexities['big_o']:
        big_o_match = re.search(r'o\s*\(([^)]+)\)', text_lower)
        if big_o_match:
            complexities['big_o'] = f"O({big_o_match.group(1).strip()})"
    
    if not complexities['omega']:
        omega_match = re.search(r'[ω|omega]\s*\(([^)]+)\)', text_lower)
        if omega_match:
            complexities['omega'] = f"Ω({omega_match.group(1).strip()})"
    
    if not complexities['theta']:
        theta_match = re.search(r'[θ|theta]\s*\(([^)]+)\)', text_lower)
        if theta_match:
            complexities['theta'] = f"Θ({theta_match.group(1).strip()})"
    
    return complexities


def normalize_complexity(complexity: str) -> str:
    """Normaliza una notación de complejidad para comparación."""
    if not complexity or complexity == 'No detectado':
        return ""
    
    # Remover espacios y convertir a minúsculas
    norm = complexity.lower().replace(' ', '')
    
    # Extraer solo la función, sin la notación
    norm = norm.replace('o(', '').replace('ω(', '').replace('θ(', '')
    norm = norm.replace('omega(', '').replace('theta(', '').replace(')', '')
    
    # Normalizar exponentes
    norm = norm.replace('²', '^2').replace('³', '^3')
    norm = norm.replace('n^2', 'n2').replace('n^3', 'n3')
    
    # Normalizar logaritmos
    norm = norm.replace('log(n)', 'logn').replace('log n', 'logn')
    norm = norm.replace('lgn', 'logn').replace('lg(n)', 'logn')
    
    # Normalizar multiplicaciones
    norm = norm.replace('*', '').replace('·', '').replace('×', '')
    
    return norm


def are_equivalent(complexity1: str, complexity2: str) -> bool:
    """Verifica si dos complejidades son matemáticamente equivalentes."""
    if not complexity1 or not complexity2:
        return False
    
    # Comparación directa
    if complexity1 == complexity2:
        return True
    
    # Equivalencias conocidas
    equivalences = {
        'n2': ['nn', 'nsquared', 'n^2'],
        'n3': ['nnn', 'ncubed', 'n^3'],
        'logn': ['log(n)', 'lgn'],
        'nlogn': ['nlog(n)', 'n*logn'],
        '1': ['constant', 'k'],
    }
    
    for standard, variants in equivalences.items():
        if complexity1 in [standard] + variants and complexity2 in [standard] + variants:
            return True
    
    return False


def analyze_discrepancy(auto: str, gemini: str, code: str) -> str:
    """Analiza la causa de una discrepancia entre automático y Gemini."""
    if not gemini or gemini == 'N/A':
        return "Gemini no proporcionó esta notación"
    
    # Contar loops
    loop_count = code.lower().count('for') + code.lower().count('para')
    loop_count += code.lower().count('while') + code.lower().count('mientras')
    
    # Detectar anidación
    nested = _count_nested_depth(code)
    
    auto_norm = normalize_complexity(auto)
    gemini_norm = normalize_complexity(gemini)
    
    # Análisis específico
    if 'logn' in gemini_norm and 'logn' not in auto_norm:
        if '/2' in code or '// 2' in code:
            return "Algoritmo tiene patrón logarítmico (/2) que no fue detectado correctamente"
        return "Gemini identificó complejidad logarítmica no detectada por el sistema"
    
    if nested >= 2 and 'n2' not in auto_norm:
        return f"Código tiene {nested} loops anidados pero el sistema no detectó O(n²)"
    
    if loop_count == 0 and auto_norm != '1':
        return "Sin loops, debería ser O(1)"
    
    if 'n2' in auto_norm and nested < 2:
        return "Sistema detectó O(n²) pero no hay suficientes loops anidados"
    
    return "Diferencia en interpretación del algoritmo"


def _count_nested_depth(code: str) -> int:
    """Cuenta la profundidad máxima de loops anidados."""
    lines = code.split('\n')
    depth = 0
    max_depth = 0
    
    for line in lines:
        line_lower = line.strip().lower()
        if re.search(r'\b(for|para|while|mientras)\b.*\b(hacer|do)\b', line_lower):
            depth += 1
            max_depth = max(max_depth, depth)
        elif re.search(r'\b(fin|end)\b', line_lower):
            depth = max(0, depth - 1)
    
    return max_depth


def generate_recommendations(issues: list, complexity, gemini_complexities: dict, code: str) -> list:
    """Genera recomendaciones para mejorar el análisis."""
    recommendations = []
    
    if not issues:
        return recommendations
    
    for issue in issues:
        if issue['reason'] == "Gemini no proporcionó esta notación":
            recommendations.append(
                f"Ajustar prompt de Gemini para que siempre proporcione {issue['type']}"
            )
        elif "no fue detectado correctamente" in issue['reason']:
            recommendations.append(
                f"Mejorar detección de patrones logarítmicos en ComplexityAnalyzer"
            )
        elif "loops anidados" in issue['reason']:
            recommendations.append(
                f"Verificar conteo de loops anidados en Parser.analyze_by_line()"
            )
        elif "Sin loops, debería ser O(1)" in issue['reason']:
            recommendations.append(
                f"ComplexityAnalyzer debe retornar O(1) cuando no hay loops"
            )
    
    return recommendations


def test_multiple_algorithms():
    """Prueba diagnóstica con múltiples algoritmos."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          DIAGNÓSTICO MÚLTIPLE - ANÁLISIS DE PUNTUACIONES BAJAS      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Cargar ejemplos desde directorio
    examples_dir = Path(__file__).parent.parent / 'pseudocode'
    
    if not examples_dir.exists():
        print(f" Directorio no encontrado: {examples_dir}")
        return
    
    txt_files = list(examples_dir.glob('*.txt'))[:3]  # Solo primeros 3 para diagnóstico
    
    if not txt_files:
        print(f" No se encontraron archivos .txt en {examples_dir}")
        return
    
    for txt_file in txt_files:
        try:
            code = txt_file.read_text(encoding='utf-8')
            diagnose_algorithm(code, txt_file.stem)
            
            input("\n  Presiona Enter para continuar al siguiente algoritmo...")
        except Exception as e:
            print(f"❌ Error procesando {txt_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Modo archivo
        file_path = sys.argv[1]
        code = Path(file_path).read_text(encoding='utf-8')
        diagnose_algorithm(code, Path(file_path).stem)
    else:
        # Modo interactivo
        print("Selecciona modo:")
        print("1. Diagnóstico de múltiples algoritmos")
        print("2. Código manual")
        
        choice = input("\nOpción (1-2): ").strip()
        
        if choice == "1":
            test_multiple_algorithms()
        elif choice == "2":
            print("\nIngresa pseudocódigo (línea vacía para terminar):")
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
            name = input("\nNombre del algoritmo: ").strip() or "manual"
            diagnose_algorithm(code, name)
        else:
            print("Opción inválida")