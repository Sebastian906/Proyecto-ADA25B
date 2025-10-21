"""
Módulo de Validación y Comparación de Análisis
Compara resultados automáticos con respuestas de LLM (Gemini)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import json
from datetime import datetime
from core.complexity import ComplexityAnalyzer, ComplexityResult
from core.patterns import PatternRecognizer
from core.parser import PseudocodeParser
from llm.integration import ask_gemini

@dataclass
class ValidationResult:
    """Resultado de la validación comparativa"""
    timestamp: str
    algorithm_name: str
    
    # Complejidades
    complexity_match: Dict[str, bool]
    complexity_differences: List[str]
    
    # Patrones
    pattern_match: Dict[str, Any]
    pattern_differences: List[str]
    
    # Análisis línea por línea
    line_analysis_match: float  # Porcentaje de coincidencia
    line_differences: List[Dict[str, Any]]
    
    # Ecuación de recurrencia
    recurrence_match: bool
    recurrence_comparison: Dict[str, str]
    
    # Resumen general
    overall_confidence: float
    gemini_insights: List[str]
    recommendations: List[str]
    
    # Metadata
    auto_analysis: Dict[str, Any] = field(default_factory=dict)
    gemini_analysis: Dict[str, Any] = field(default_factory=dict)

class ComplexityValidator:
    """Valida y compara análisis de complejidad"""
    
    def __init__(self):
        self.complexity_patterns = {
            'O(1)': ['constante', 'constant', 'o(1)', 'O(1)'],
            'O(log n)': ['logarítmico', 'logarithmic', 'log n', 'log(n)', 'O(log n)'],
            'O(n)': ['lineal', 'linear', 'o(n)', 'O(n)'],
            'O(n log n)': ['n log n', 'nlogn', 'linearítmico', 'O(n log n)'],
            'O(n²)': ['cuadrático', 'quadratic', 'n²', 'n cuadrado', 'O(n²)', 'O(n^2)'],
            'O(n³)': ['cúbico', 'cubic', 'n³', 'n cubo', 'O(n³)', 'O(n^3)'],
            'O(2^n)': ['exponencial', 'exponential', '2^n', 'O(2^n)'],
        }
    
    def compare_complexities(self, auto_result: ComplexityResult, 
                            gemini_text: str) -> Tuple[Dict[str, bool], List[str]]:
        """
        Compara complejidades automáticas con análisis de Gemini
        
        Returns:
            Tuple[Dict[str, bool], List[str]]: (coincidencias, diferencias)
        """
        matches = {
            'big_o': False,
            'omega': False,
            'theta': False
        }
        differences = []
        
        gemini_lower = gemini_text.lower()
        
        # Comparar Big-O
        auto_big_o = auto_result.big_o
        if self._find_complexity_in_text(auto_big_o, gemini_lower):
            matches['big_o'] = True
        else:
            gemini_big_o = self._extract_complexity_from_text(gemini_lower, 'peor caso')
            differences.append(
                f"Big-O: Automático={auto_big_o}, Gemini={gemini_big_o or 'No detectado'}"
            )
        
        # Comparar Omega
        auto_omega = auto_result.omega
        if self._find_complexity_in_text(auto_omega, gemini_lower):
            matches['omega'] = True
        else:
            gemini_omega = self._extract_complexity_from_text(gemini_lower, 'mejor caso')
            differences.append(
                f"Omega: Automático={auto_omega}, Gemini={gemini_omega or 'No detectado'}"
            )
        
        # Comparar Theta
        auto_theta = auto_result.theta
        if self._find_complexity_in_text(auto_theta, gemini_lower):
            matches['theta'] = True
        else:
            gemini_theta = self._extract_complexity_from_text(gemini_lower, 'caso promedio')
            differences.append(
                f"Theta: Automático={auto_theta}, Gemini={gemini_theta or 'No detectado'}"
            )
        
        return matches, differences
    
    def _find_complexity_in_text(self, complexity: str, text: str) -> bool:
        """Busca una complejidad específica en el texto con mayor robustez"""
        if not complexity:
            return False
        
        # Normalizar complejidad
        clean_complexity = complexity.replace('O', '').replace('Ω', '').replace('Θ', '').strip()
        clean_complexity = clean_complexity.replace('(', '').replace(')', '').strip()
        clean_complexity = clean_complexity.replace('^', '').replace('²', '2').replace('³', '3')
        
        # Normalizar texto
        text_lower = text.lower()
        text_lower = text_lower.replace('^', '').replace('²', '2').replace('³', '3')
        
        # Buscar patrones equivalentes con puntuación de coincidencia
        if clean_complexity in self.complexity_patterns:
            patterns = self.complexity_patterns[clean_complexity]
            pattern_scores = [
                sum(1 for c in pattern.lower() if c in text_lower) / len(pattern)
                for pattern in patterns
            ]
            if max(pattern_scores, default=0) > 0.8:  # 80% de coincidencia
                return True
        
        # Buscar variaciones comunes con análisis contextual
        variations = [
            complexity.lower(),
            clean_complexity.lower(),
            f"o({clean_complexity})",
            f"O({clean_complexity})",
            f"complejidad {clean_complexity}",
            clean_complexity.replace(' ', ''),
            # Agregar variaciones numéricas
            clean_complexity.replace('n', 'N'),
            clean_complexity.replace('log', 'lg'),
            clean_complexity.replace('2^n', 'exponencial'),
            clean_complexity.replace('n^2', 'n*n'),
        ]
        
        # Buscar en contexto de palabras clave
        complexity_contexts = [
            'complejidad', 'complexity',
            'tiempo', 'time',
            'costo', 'cost',
            'orden', 'order',
            'asintótico', 'asymptotic'
        ]
        
        for context in complexity_contexts:
            if context in text_lower:
                # Buscar variaciones cerca del contexto (+/- 50 caracteres)
                idx = text_lower.find(context)
                search_text = text_lower[max(0, idx-50):min(len(text_lower), idx+50)]
                if any(var in search_text for var in variations):
                    return True
        
        return any(var in text_lower for var in variations)
    
    def _extract_complexity_from_text(self, text: str, context: str) -> Optional[str]:
        """Extrae complejidad del texto de Gemini"""
        # Buscar patrones cerca del contexto
        context_pattern = re.escape(context)
        
        # Buscar O(...), Ω(...), Θ(...)
        pattern = rf'{context_pattern}.*?(O|Ω|Θ)\s*\(\s*([^)]+)\s*\)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            notation = match.group(1)
            complexity = match.group(2).strip()
            return f"{notation}({complexity})"
        
        # Buscar descripciones textuales
        for std_complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text and context.lower() in text:
                    # Verificar proximidad
                    context_idx = text.find(context.lower())
                    pattern_idx = text.find(pattern.lower())
                    if abs(context_idx - pattern_idx) < 100:
                        return std_complexity
        
        return None

class PatternValidator:
    """Valida y compara patrones algorítmicos"""
    
    def __init__(self):
        self.pattern_keywords = {
            'divide_y_conquista': [
                'divide y conquista', 'divide and conquer', 'dividir', 'merge',
                'partir problema', 'combinar soluciones', 'mergesort', 'quicksort'
            ],
            'busqueda_binaria': [
                'búsqueda binaria', 'binary search', 'binaria', 'dicotómica',
                'búsqueda por mitades', 'divide intervalo', 'búsqueda logarítmica'
            ],
            'programacion_dinamica': [
                'programación dinámica', 'dynamic programming', 'memoización',
                'subproblemas', 'tabla dp', 'overlapping subproblems',
                'optimal substructure', 'bottom-up', 'top-down'
            ],
            'algoritmo_voraz': [
                'greedy', 'voraz', 'ávido', 'codicioso', 'óptimo local',
                'selección local', 'elección golosa', 'locally optimal'
            ],
            'backtracking': [
                'backtracking', 'retroceso', 'vuelta atrás', 'recursión con retorno',
                'exploración exhaustiva', 'prueba y error', 'depth-first search'
            ],
            'fuerza_bruta': [
                'fuerza bruta', 'brute force', 'exhaustiva', 'todas las combinaciones',
                'todas las posibilidades', 'exhaustive search'
            ],
            'recursion': [
                'recursión', 'recursion', 'llamada recursiva', 'caso base',
                'recursive call', 'base case', 'recursive solution'
            ],
            'ordenamiento': [
                'ordenamiento', 'sorting', 'clasificación', 'ordenar array',
                'bubble sort', 'insertion sort', 'selection sort'
            ],
            'dos_punteros': [
                'two pointers', 'dos punteros', 'ventana deslizante',
                'sliding window', 'pointer technique'
            ]
        }
    
    def compare_patterns(self, auto_patterns: List[Dict], 
                        gemini_text: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Compara patrones detectados automáticamente con Gemini
        
        Returns:
            Tuple[Dict, List]: (estadísticas de coincidencia, diferencias)
        """
        gemini_lower = gemini_text.lower()
        
        matches = {
            'detected_auto': len(auto_patterns),
            'confirmed_by_gemini': 0,
            'additional_from_gemini': [],
            'matched_patterns': []
        }
        differences = []
        
        # Verificar patrones automáticos
        for pattern in auto_patterns:
            pattern_name = pattern.get('name', '')
            pattern_desc = pattern.get('description', '')
            
            if self._pattern_in_text(pattern_name, gemini_lower):
                matches['confirmed_by_gemini'] += 1
                matches['matched_patterns'].append(pattern_name)
            else:
                differences.append(
                    f"Patrón '{pattern_desc}' detectado automáticamente pero no confirmado por Gemini"
                )
        
        # Buscar patrones adicionales en Gemini
        for pattern_key, keywords in self.pattern_keywords.items():
            if any(kw in gemini_lower for kw in keywords):
                # Verificar si ya fue detectado automáticamente
                if not any(p.get('name') == pattern_key for p in auto_patterns):
                    matches['additional_from_gemini'].append(pattern_key)
                    differences.append(
                        f"Patrón '{pattern_key}' mencionado por Gemini pero no detectado automáticamente"
                    )
        
        return matches, differences
    
    def _pattern_in_text(self, pattern_name: str, text: str) -> bool:
        """Verifica si un patrón está mencionado en el texto"""
        if pattern_name in self.pattern_keywords:
            keywords = self.pattern_keywords[pattern_name]
            return any(kw in text for kw in keywords)
        
        # Búsqueda por nombre directo
        return pattern_name.lower() in text

class LineAnalysisValidator:
    """Valida análisis línea por línea"""
    
    def compare_line_analysis(self, auto_lines: List[Dict], 
                                gemini_text: str) -> Tuple[float, List[Dict]]:
        """
        Compara análisis línea por línea
        
        Returns:
            Tuple[float, List]: (porcentaje de coincidencia, diferencias detalladas)
        """
        differences: List[Dict[str, Any]] = []
        matches: float = 0.0

        total_lines = len([l for l in auto_lines if l.get('type') != 'summary'])

        if total_lines == 0:
            return 0.0, []

        gemini_lower = gemini_text.lower()

        for line_info in auto_lines:
            if line_info.get('type') == 'summary':
                continue

            line_no = line_info.get('line', 0)
            time_cost = line_info.get('time_cost', '')
            exec_count = line_info.get('exec_count', '')

            # Verificar si Gemini menciona esta línea (aceptar varias formas)
            line_patterns = [
                f"línea {line_no}",
                f"line {line_no}",
                f"paso {line_no}",
                f"línea:{line_no}",
                f"line:{line_no}",
                f"l{line_no}",
                f"#{line_no}",
            ]

            mentioned = any(pattern in gemini_lower for pattern in line_patterns)

            if mentioned:
                # Si tenemos un costo automático y Gemini lo menciona cerca, contar match completo
                if time_cost and self._complexity_near_line(time_cost, line_no, gemini_text):
                    matches += 1.0
                else:
                    # Si Gemini menciona la línea pero no encuentra el costo exacto, contar match parcial
                    matches += 0.5
                    differences.append({
                        'line': line_no,
                        'auto_cost': time_cost,
                        'gemini_mentioned': True,
                        'costs_match': False
                    })
            else:
                differences.append({
                    'line': line_no,
                    'auto_cost': time_cost,
                    'gemini_mentioned': False,
                    'costs_match': None
                })

        match_percentage = (matches / total_lines) * 100 if total_lines > 0 else 0
        return match_percentage, differences
    
    def _complexity_near_line(self, complexity: str, line_no: int, text: str) -> bool:
        """Verifica si una complejidad está mencionada cerca de una línea con análisis contextual mejorado"""
        text_lower = text.lower()
        
        # Patrones de línea expandidos
        line_patterns = [
            f"línea {line_no}",
            f"line {line_no}",
            f"paso {line_no}",
            f"instrucción {line_no}",
            f"statement {line_no}",
            f"l{line_no}",
            f"#{line_no}",
        ]
        
        # Palabras clave de complejidad
        complexity_keywords = [
            'complejidad', 'complexity',
            'costo', 'cost',
            'tiempo', 'time',
            'operaciones', 'operations',
            'iteraciones', 'iterations'
        ]
        
        # Normalizar complejidad
        clean_complexity = complexity.lower()
        clean_complexity = clean_complexity.replace('o', '').replace('(', '').replace(')', '').strip()
        clean_complexity = clean_complexity.replace('^', '').replace('²', '2').replace('³', '3')
        
        # Variaciones de complejidad
        complexity_variations = [
            clean_complexity,
            f"o({clean_complexity})",
            f"O({clean_complexity})",
            clean_complexity.replace('n', 'N'),
            clean_complexity.replace('log', 'lg'),
            complexity.lower()
        ]
        
        for pattern in line_patterns:
            idx = text_lower.find(pattern)
            if idx != -1:
                # Buscar en un contexto más amplio (+/- 150 caracteres)
                start_idx = max(0, idx - 150)
                end_idx = min(len(text_lower), idx + 150)
                context = text_lower[start_idx:end_idx]
                
                # Verificar si hay palabras clave de complejidad cerca
                has_complexity_context = any(keyword in context for keyword in complexity_keywords)
                
                if has_complexity_context:
                    # Buscar variaciones de complejidad
                    for var in complexity_variations:
                        if var in context:
                            # Calcular distancia entre la línea y la complejidad
                            line_pos = context.find(pattern)
                            complexity_pos = context.find(var)
                            distance = abs(line_pos - complexity_pos)
                            
                            # Si está lo suficientemente cerca (dentro de 100 caracteres)
                            if distance <= 100:
                                return True
                
                # Buscar patrones numéricos específicos
                if clean_complexity == '1':
                    if 'constante' in context or 'constant' in context:
                        return True
                elif clean_complexity == 'n':
                    if 'lineal' in context or 'linear' in context:
                        return True
                elif clean_complexity == 'n2' or clean_complexity == 'n²':
                    if 'cuadrático' in context or 'quadratic' in context:
                        return True
        
        return False

class RecurrenceValidator:
    """Valida ecuaciones de recurrencia"""
    
    def compare_recurrence(self, auto_recurrence: Optional[str], 
                            gemini_text: str) -> Tuple[bool, Dict[str, str]]:
        """
        Compara ecuaciones de recurrencia
        
        Returns:
            Tuple[bool, Dict]: (coincide, comparación detallada)
        """
        comparison = {
            'auto': auto_recurrence or 'No detectada',
            'gemini': 'No detectada'
        }
        
        if not auto_recurrence:
            return False, comparison
        
        # Buscar ecuación en texto de Gemini
        recurrence_patterns = [
            r'T\(n\)\s*=\s*([^.]+)',
            r'recurrencia:\s*([^.]+)',
            r'ecuación:\s*([^.]+)',
        ]
        
        gemini_recurrence = None
        for pattern in recurrence_patterns:
            match = re.search(pattern, gemini_text, re.IGNORECASE)
            if match:
                gemini_recurrence = match.group(1).strip()
                comparison['gemini'] = gemini_recurrence
                break
        
        if not gemini_recurrence:
            return False, comparison
        
        # Normalizar y comparar
        auto_normalized = self._normalize_recurrence(auto_recurrence)
        gemini_normalized = self._normalize_recurrence(gemini_recurrence)
        
        match = auto_normalized == gemini_normalized
        return match, comparison
    
    def _normalize_recurrence(self, recurrence: str) -> str:
        """Normaliza ecuación de recurrencia para comparación más robusta"""
        if not recurrence:
            return ""
            
        # Normalización básica
        normalized = recurrence.lower()
        normalized = re.sub(r'\s+', '', normalized)
        
        # Normalizar operadores
        normalized = normalized.replace('*', '·').replace('x', '·')
        normalized = normalized.replace('×', '·').replace('⋅', '·')
        
        # Normalizar paréntesis y símbolos
        normalized = normalized.replace('{', '(').replace('}', ')')
        normalized = normalized.replace('[', '(').replace(']', ')')
        
        # Normalizar notación de funciones
        normalized = re.sub(r't\(([^)]+)\)', r'T(\1)', normalized)  # t(n) -> T(n)
        normalized = re.sub(r'rec\(([^)]+)\)', r'T(\1)', normalized)  # rec(n) -> T(n)
        
        # Normalizar exponentes
        normalized = normalized.replace('^', '**')
        normalized = re.sub(r'(\d+)n', r'\1·n', normalized)  # 2n -> 2·n
        
        # Normalizar operaciones comunes
        normalized = re.sub(r'n/2', r'n·0.5', normalized)
        normalized = normalized.replace('2^n', 'pow(2,n)')
        
        # Normalizar términos especiales
        normalized = re.sub(r'log(\d*)', r'log_\1', normalized)  # log2 -> log_2
        normalized = normalized.replace('ln', 'log_e')
        
        # Simplificar expresiones comunes
        normal_forms = {
            'n·n': 'n^2',
            'n·n·n': 'n^3',
            'n·log_2(n)': 'n·log(n)',
            'n·log_e(n)': 'n·log(n)',
        }
        
        for original, simplified in normal_forms.items():
            normalized = normalized.replace(original, simplified)
            
        return normalized

class ValidationOrchestrator:
    """Orquestador principal de validación"""
    
    def __init__(self):
        self.complexity_validator = ComplexityValidator()
        self.pattern_validator = PatternValidator()
        self.line_validator = LineAnalysisValidator()
        self.recurrence_validator = RecurrenceValidator()
        
        self.complexity_analyzer = ComplexityAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.parser = PseudocodeParser()
    
    def validate_algorithm(self, code: str, algorithm_name: str = "algorithm") -> ValidationResult:
        """
        Validación completa: automático vs Gemini
        
        Args:
            code: Pseudocódigo del algoritmo
            algorithm_name: Nombre identificador del algoritmo
            
        Returns:
            ValidationResult: Resultado completo de la validación
        """
        timestamp = datetime.now().isoformat()
        
        # 1. Análisis automático
        print("Ejecutando análisis automático...")
        auto_complexity = self.complexity_analyzer.analyze(code)
        auto_patterns = self.pattern_recognizer.analyze(code)
        auto_lines = self.parser.analyze_by_line(code)
        
        auto_recurrence = None
        for item in auto_lines:
            if item.get('type') == 'summary' and 'recurrence' in item:
                auto_recurrence = item.get('recurrence')
                break
        
        # 2. Consultar Gemini
        print("Consultando Gemini para validación...")
        gemini_prompt = self._build_validation_prompt(code)
        gemini_response = ask_gemini(gemini_prompt)
        
        # 3. Comparaciones
        print("Comparando resultados...")
        
        complexity_match, complexity_diffs = self.complexity_validator.compare_complexities(
            auto_complexity, gemini_response
        )
        
        pattern_match, pattern_diffs = self.pattern_validator.compare_patterns(
            auto_patterns, gemini_response
        )
        
        line_match_pct, line_diffs = self.line_validator.compare_line_analysis(
            auto_lines, gemini_response
        )
        
        recurrence_match, recurrence_comp = self.recurrence_validator.compare_recurrence(
            auto_recurrence, gemini_response
        )
        
        # 4. Extraer insights de Gemini
        gemini_insights = self._extract_insights(gemini_response)
        
        # 5. Generar recomendaciones
        recommendations = self._generate_recommendations(
            complexity_match, pattern_match, line_match_pct, recurrence_match
        )
        
        # 6. Calcular confianza general
        overall_confidence = self._calculate_confidence(
            complexity_match, pattern_match, line_match_pct, recurrence_match
        )
        
        # 7. Construir resultado
        result = ValidationResult(
            timestamp=timestamp,
            algorithm_name=algorithm_name,
            complexity_match=complexity_match,
            complexity_differences=complexity_diffs,
            pattern_match=pattern_match,
            pattern_differences=pattern_diffs,
            line_analysis_match=line_match_pct,
            line_differences=line_diffs,
            recurrence_match=recurrence_match,
            recurrence_comparison=recurrence_comp,
            overall_confidence=overall_confidence,
            gemini_insights=gemini_insights,
            recommendations=recommendations,
            auto_analysis={
                'complexity': {
                    'big_o': auto_complexity.big_o,
                    'omega': auto_complexity.omega,
                    'theta': auto_complexity.theta,
                    'explanation': auto_complexity.explanation
                },
                'patterns': auto_patterns,
                'recurrence': auto_recurrence
            },
            gemini_analysis={
                'full_response': gemini_response[:500] + '...' if len(gemini_response) > 500 else gemini_response
            }
        )
        
        return result
    
    def _build_validation_prompt(self, code: str) -> str:
        """Construye prompt optimizado para validación"""
        prompt = f"""Analiza el siguiente algoritmo en pseudocódigo y proporciona:

1. Complejidad temporal en notación Big-O (peor caso), Omega (mejor caso) y Theta (caso promedio)
2. Patrones algorítmicos identificados (divide y conquista, búsqueda binaria, programación dinámica, etc.)
3. Ecuación de recurrencia si aplica
4. Análisis línea por línea con costos temporales
5. Observaciones adicionales sobre optimización o mejoras

Pseudocódigo:
```
{code}
```

Por favor, sé específico con las notaciones asintóticas y menciona explícitamente cada patrón detectado."""
        
        return prompt
    
    def _extract_insights(self, gemini_text: str) -> List[str]:
        """Extrae insights clave de la respuesta de Gemini"""
        insights = []
        
        # Buscar secciones clave
        sections = {
            'optimización': ['optimización', 'optimization', 'mejorar', 'improve'],
            'observación': ['observación', 'observation', 'nota', 'note'],
            'advertencia': ['advertencia', 'warning', 'cuidado', 'caution'],
        }
        
        for section_name, keywords in sections.items():
            for keyword in keywords:
                if keyword in gemini_text.lower():
                    # Extraer contexto alrededor de la palabra clave
                    idx = gemini_text.lower().find(keyword)
                    context = gemini_text[idx:min(idx+200, len(gemini_text))]
                    # Extraer hasta el primer punto
                    sentence = context.split('.')[0]
                    if sentence and len(sentence) > 10:
                        insights.append(f"[{section_name.upper()}] {sentence.strip()}")
                        break
        
        return insights
    
    def _generate_recommendations(self, complexity_match: Dict, pattern_match: Dict,
                                    line_match: float, recurrence_match: bool) -> List[str]:
        """Genera recomendaciones basadas en las comparaciones"""
        recommendations = []
        
        # Recomendaciones de complejidad
        matched_complexities = sum(complexity_match.values())
        if matched_complexities == 3:
            recommendations.append("✓ Análisis de complejidad completamente validado por Gemini")
        elif matched_complexities >= 1:
            recommendations.append("⚠ Revisar discrepancias en algunas notaciones de complejidad")
        else:
            recommendations.append("✗ Divergencia significativa en análisis de complejidad - revisar algoritmo")
        
        # Recomendaciones de patrones
        if pattern_match.get('confirmed_by_gemini', 0) == pattern_match.get('detected_auto', 0):
            recommendations.append("✓ Todos los patrones algorítmicos confirmados")
        elif pattern_match.get('additional_from_gemini'):
            recommendations.append(
                f"⚠ Gemini identificó patrones adicionales: {', '.join(pattern_match['additional_from_gemini'])}"
            )
        
        # Recomendaciones de análisis línea por línea
        if line_match >= 80:
            recommendations.append("✓ Alto nivel de concordancia en análisis línea por línea")
        elif line_match >= 50:
            recommendations.append("⚠ Concordancia moderada - verificar costos de ejecución específicos")
        else:
            recommendations.append("✗ Baja concordancia en análisis detallado - revisar implementación")
        
        # Recomendaciones de recurrencia
        if recurrence_match:
            recommendations.append("✓ Ecuación de recurrencia validada")
        else:
            recommendations.append("⚠ Revisar ecuación de recurrencia - puede haber diferencias")
        
        return recommendations
    
    def _calculate_confidence(self, complexity_match: Dict, pattern_match: Dict,
                                line_match: float, recurrence_match: bool) -> float:
        """Calcula nivel de confianza general (0-100) con pesos ajustados y factores adicionales"""
        weights = {
            'complexity': 0.40,  # Mayor peso a la complejidad
            'patterns': 0.30,    # Incremento en patrones
            'lines': 0.20,       # Reducción en líneas
            'recurrence': 0.10   # Reducción en recurrencia
        }
        
        # Score de complejidad con bonificación por coincidencias completas
        matches_count = sum(complexity_match.values())
        complexity_base = (matches_count / 3) * 100
        complexity_bonus = 10 if matches_count == 3 else 0  # Bonificación por coincidencia total
        complexity_score = min(100, complexity_base + complexity_bonus)
        
        # Score de patrones con factor de precisión
        detected = pattern_match.get('detected_auto', 0)
        confirmed = pattern_match.get('confirmed_by_gemini', 0)
        additional = len(pattern_match.get('additional_from_gemini', []))
        
        if detected == 0:
            pattern_score = 100 if additional == 0 else 70  # Penalización si Gemini encuentra patrones no detectados
        else:
            precision = confirmed / detected
            recall = confirmed / (confirmed + additional) if (confirmed + additional) > 0 else 1
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            pattern_score = f1_score * 100
        
        # Score de líneas con umbral mínimo
        line_score = max(line_match, 50) if line_match >= 30 else line_match  # Umbral mínimo de confianza
        
        # Score de recurrencia con análisis de complejidad
        recurrence_base = 100 if recurrence_match else 0
        recurrence_context = 20 if matches_count >= 2 else 0  # Bonus por contexto de complejidad
        recurrence_score = min(100, recurrence_base + recurrence_context)
        
        # Promedio ponderado con factor de consistencia
        raw_confidence = (
            complexity_score * weights['complexity'] +
            pattern_score * weights['patterns'] +
            line_score * weights['lines'] +
            recurrence_score * weights['recurrence']
        )
        
        # Factor de consistencia basado en la varianza de los scores
        scores = [complexity_score, pattern_score, line_score, recurrence_score]
        variance = sum((s - sum(scores)/4) ** 2 for s in scores) / 4
        consistency_factor = 1 - (variance / 10000)  # Normalizado a 1
        
        final_confidence = raw_confidence * consistency_factor
        
        return round(max(min(final_confidence, 100), 0), 2)
    
    def export_validation_report(self, result: ValidationResult, filepath: str = "validation_report.json"):
        """Exporta reporte de validación a JSON"""
        report = {
            'metadata': {
                'timestamp': result.timestamp,
                'algorithm': result.algorithm_name,
                'confidence': result.overall_confidence
            },
            'complexity_analysis': {
                'matches': result.complexity_match,
                'differences': result.complexity_differences,
                'auto': result.auto_analysis.get('complexity')
            },
            'pattern_analysis': {
                'matches': result.pattern_match,
                'differences': result.pattern_differences,
                'auto': result.auto_analysis.get('patterns')
            },
            'line_analysis': {
                'match_percentage': result.line_analysis_match,
                'differences': result.line_differences
            },
            'recurrence_analysis': {
                'match': result.recurrence_match,
                'comparison': result.recurrence_comparison
            },
            'insights': result.gemini_insights,
            'recommendations': result.recommendations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Reporte de validación exportado: {filepath}")
    
    def print_validation_summary(self, result: ValidationResult):
        """Imprime resumen legible de la validación"""
        print("\n" + "="*70)
        print(f"REPORTE DE VALIDACIÓN: {result.algorithm_name}")
        print("="*70)
        print(f"Timestamp: {result.timestamp}")
        print(f"Confianza General: {result.overall_confidence:.2f}%")
        print("\n" + "-"*70)
        
        print("\n COMPLEJIDAD ALGORÍTMICA:")
        for notation, matched in result.complexity_match.items():
            status = "✓" if matched else "✗"
            print(f"  {status} {notation.upper()}: {'Validado' if matched else 'Discrepancia'}")
        
        if result.complexity_differences:
            print("\n  Diferencias detectadas:")
            for diff in result.complexity_differences:
                print(f"    - {diff}")
        
        print("\n PATRONES ALGORÍTMICOS:")
        print(f"  Detectados automáticamente: {result.pattern_match.get('detected_auto', 0)}")
        print(f"  Confirmados por Gemini: {result.pattern_match.get('confirmed_by_gemini', 0)}")
        
        if result.pattern_match.get('matched_patterns'):
            print(f"  Patrones validados: {', '.join(result.pattern_match['matched_patterns'])}")
        
        if result.pattern_differences:
            print("\n  Diferencias:")
            for diff in result.pattern_differences[:3]:  # Mostrar máximo 3
                print(f"    - {diff}")
        
        print(f"\n ANÁLISIS LÍNEA POR LÍNEA:")
        print(f"  Coincidencia: {result.line_analysis_match:.2f}%")
        
        print(f"\n ECUACIÓN DE RECURRENCIA:")
        print(f"  {'✓ Validada' if result.recurrence_match else '✗ Discrepancia'}")
        print(f"  Automático: {result.recurrence_comparison.get('auto')}")
        print(f"  Gemini: {result.recurrence_comparison.get('gemini')}")
        
        print("\n INSIGHTS DE GEMINI:")
        for insight in result.gemini_insights[:3]:
            print(f"  • {insight}")
        
        print("\n RECOMENDACIONES:")
        for rec in result.recommendations:
            print(f"  {rec}")
        
        print("\n" + "="*70 + "\n")


# Función auxiliar para uso rápido
def validate_algorithm(code: str, algorithm_name: str = "algorithm") -> ValidationResult:
    """
    Función de conveniencia para validación rápida
    
    Args:
        code: Pseudocódigo del algoritmo
        algorithm_name: Nombre del algoritmo
        
    Returns:
        ValidationResult: Resultado completo de validación
    """
    orchestrator = ValidationOrchestrator()
    result = orchestrator.validate_algorithm(code, algorithm_name)
    orchestrator.print_validation_summary(result)
    return result


if __name__ == "__main__":
    # Ejemplo de uso
    sample_code = """
    PARA i DESDE 0 HASTA n-1 HACER
        PARA j DESDE 0 HASTA n-1 HACER
            ESCRIBIR arreglo[i][j]
        FIN PARA
    FIN PARA
    """
    
    result = validate_algorithm(sample_code, "nested_loops_example")
    
    # Exportar reporte
    orchestrator = ValidationOrchestrator()
    orchestrator.export_validation_report(result, "validation_report.json")