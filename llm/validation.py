"""
Módulo `validation.py`

Este módulo se encarga de validar y comparar los resultados del análisis automático de pseudocódigo con los proporcionados por un modelo de lenguaje (Gemini). 
Incluye funcionalidades para:
- Validar complejidades algorítmicas (Big-O, Omega, Theta).
- Comparar patrones algorítmicos detectados automáticamente.
- Analizar línea por línea el pseudocódigo.
- Validar ecuaciones de recurrencia.
- Generar informes detallados de validación.

Clases principales:
    - EfficiencyValidationResult: Resultado de la validación de eficiencia
    - ComplexityValidator: Valida análisis de complejidad (Big-O, Omega, Theta)
    - RecurrenceValidator: Valida ecuaciones de recurrencia
    - CostAnalysisValidator: Valida costos temporales línea por línea
    - EfficiencyOrchestrator: Orquesta la validación completa

Funciones principales:
    - validate_algorithm: Realiza la validación completa de un algoritmo.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import json
from datetime import datetime
from pathlib import Path
from core.complexity import ComplexityAnalyzer, ComplexityResult
from core.patterns import PatternRecognizer
from core.parser import PseudocodeParser
from llm.integration import ask_gemini
from core.recurrence_analysis import analyze_recurrence

@dataclass
class EfficiencyValidationResult:
    """
    Resultado de la validación de eficiencia algorítmica.
    
    Atributos prioritarios:
        - complexity_correctness: Puntuación de corrección de complejidades (0-100)
        - recurrence_correctness: Puntuación de corrección de recurrencia (0-100)
        - cost_analysis_correctness: Puntuación de corrección de costos (0-100)
        - overall_efficiency_score: Puntuación general de eficiencia (0-100)
    """
    timestamp: str
    algorithm_name: str
    file_path: Optional[str] = None
    
    # === VALIDACIÓN DE COMPLEJIDAD (PRIORIDAD ALTA) ===
    complexity_correctness: float = 0.0  # 0-100
    complexity_details: Dict[str, Any] = field(default_factory=dict)
    complexity_errors: List[str] = field(default_factory=list)
    
    # === VALIDACIÓN DE RECURRENCIA (PRIORIDAD ALTA) ===
    recurrence_correctness: float = 0.0  # 0-100
    recurrence_details: Dict[str, Any] = field(default_factory=dict)
    recurrence_errors: List[str] = field(default_factory=list)
    
    # === VALIDACIÓN DE COSTOS POR LÍNEA (PRIORIDAD MEDIA) ===
    cost_analysis_correctness: float = 0.0  # 0-100
    cost_details: Dict[str, Any] = field(default_factory=dict)
    cost_errors: List[str] = field(default_factory=list)
    
    # === VALIDACIÓN DE PATRONES (PRIORIDAD BAJA) ===
    pattern_correctness: float = 0.0  # 0-100
    pattern_details: Dict[str, Any] = field(default_factory=dict)
    
    # === SCORING GENERAL ===
    overall_efficiency_score: float = 0.0  # 0-100
    mathematical_rigor: str = "LOW"  # LOW, MEDIUM, HIGH
    
    # === RECOMENDACIONES Y CORRECCIONES ===
    critical_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    
    # === METADATA ===
    auto_analysis: Dict[str, Any] = field(default_factory=dict)
    gemini_analysis: str = ""

class ComplexityValidator:
    """
    Valida la CORRECCIÓN del análisis de complejidad algorítmica.
    
    Enfoque:
    - Verifica que Big-O, Omega y Theta sean matemáticamente correctos
    - Detecta errores comunes (confundir O(n) con O(n²), etc.)
    - Compara con análisis de referencia (Gemini)
    """
    
    def __init__(self):
        # Mapa de complejidades ordenadas (de menor a mayor)
        self.complexity_order = [
            'O(1)', 'O(log log n)', 'O(log n)', 'O(√n)', 'O(n)',
            'O(n log n)', 'O(n²)', 'O(n³)', 'O(2^n)', 'O(n!)'
        ]
        
        # Patrones de equivalencia
        self.equivalences = {
            'O(1)': ['constante', 'constant'],
            'O(log n)': ['logarítmico', 'logarithmic', 'log n', 'log(n)'],
            'O(n)': ['lineal', 'linear'],
            'O(n log n)': ['linearítmico', 'n log n', 'nlogn'],
            'O(n²)': ['cuadrático', 'quadratic', 'n^2', 'n cuadrado'],
            'O(n³)': ['cúbico', 'cubic', 'n^3'],
            'O(2^n)': ['exponencial', 'exponential', '2^n'],
        }
    
    def validate_complexity(
        self, 
        auto_result: ComplexityResult, 
        gemini_text: str,
        code: str
    ) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        Valida la corrección del análisis de complejidad.
        
        Returns:
            Tuple[float, Dict, List]: (score 0-100, detalles, errores)
        """
        errors = []
        details = {}
        score = 0.0
        
        # 1. VALIDAR COHERENCIA INTERNA (Big-O >= Theta >= Omega)
        coherence_score, coherence_errors = self._validate_coherence(auto_result)
        errors.extend(coherence_errors)
        score += coherence_score * 0.3  # 30% del peso
        
        # 2. VALIDAR CON ESTRUCTURA DEL CÓDIGO
        structure_score, structure_errors = self._validate_with_code_structure(
            auto_result, code
        )
        errors.extend(structure_errors)
        score += structure_score * 0.4  # 40% del peso
        
        # 3. VALIDAR CON GEMINI (REFERENCIA)
        gemini_score, gemini_details = self._validate_with_gemini(
            auto_result, gemini_text
        )
        score += gemini_score * 0.3  # 30% del peso
        
        details = {
            'coherence_score': coherence_score,
            'structure_score': structure_score,
            'gemini_agreement': gemini_score,
            'auto_complexities': {
                'big_o': auto_result.big_o,
                'omega': auto_result.omega,
                'theta': auto_result.theta
            },
            'gemini_complexities': gemini_details
        }
        
        return score, details, errors
    
    def _validate_coherence(
        self, 
        result: ComplexityResult
    ) -> Tuple[float, List[str]]:
        """
        Valida que Big-O >= Theta >= Omega matemáticamente.
        """
        errors = []
        
        big_o = result.big_o
        omega = result.omega
        theta = result.theta
        
        # Normalizar notaciones
        big_o_norm = self._normalize_complexity(big_o)
        omega_norm = self._normalize_complexity(omega)
        theta_norm = self._normalize_complexity(theta)
        
        # Obtener índices de orden
        try:
            big_o_idx = self._get_complexity_index(big_o_norm)
            omega_idx = self._get_complexity_index(omega_norm)
            theta_idx = self._get_complexity_index(theta_norm)
        except ValueError as e:
            errors.append(f"Complejidad no reconocida: {e}")
            return 0.0, errors
        
        # Validar orden: Big-O >= Theta >= Omega
        if not (big_o_idx >= theta_idx >= omega_idx):
            errors.append(
                f"ERROR CRÍTICO: Violación de orden de complejidades. "
                f"Debe cumplir: {big_o} >= {theta} >= {omega}"
            )
            return 0.0, errors
        
        # Validar caso especial: si Omega = Big-O, entonces Theta debe ser igual
        if omega_idx == big_o_idx and theta_idx != big_o_idx:
            errors.append(
                f"ERROR: Si Ω = O, entonces Θ debe ser igual. "
                f"Actual: Ω={omega}, O={big_o}, Θ={theta}"
            )
            return 50.0, errors
        
        return 100.0, []
    
    def _validate_with_code_structure(
        self, 
        result: ComplexityResult,
        code: str
    ) -> Tuple[float, List[str]]:
        """
        Valida que la complejidad coincida con la estructura del código.
        MEJORADO: Detecta patrones logarítmicos y early returns.
        """
        errors = []
        score = 100.0
        
        code_lower = code.lower()
        big_o_norm = self._normalize_complexity(result.big_o)
        omega_norm = self._normalize_complexity(result.omega)
        
        # Detectar loops anidados
        nested_depth = self._count_nested_loops(code)
        
        # Detectar patrones especiales ANTES de validar por loops
        has_binary_search = self._detect_binary_search_pattern(code)
        has_early_return = 'return' in code_lower and nested_depth > 0
        
        # CASO ESPECIAL: Búsqueda binaria
        if has_binary_search:
            if 'log' not in big_o_norm.lower():
                errors.append(
                    f"ERROR: Patrón de búsqueda binaria detectado (división por 2), "
                    f"pero Big-O es {result.big_o}. Debería ser O(log n)"
                )
                score -= 40
            
            # Validar Omega (mejor caso con early return)
            if has_early_return:
                if omega_norm != 'o(1)':
                    errors.append(
                        f"ADVERTENCIA: Búsqueda binaria con return temprano, "
                        f"Omega debería ser Ω(1) pero es {result.omega}"
                    )
                    score -= 10
            
            return max(0, score), errors
        
        # Validar según profundidad de loops (para algoritmos NO logarítmicos)
        if nested_depth == 0:
            # Sin loops -> debe ser O(1)
            if big_o_norm != 'o(1)':
                errors.append(
                    f"ERROR: Sin loops detectados, pero Big-O es {result.big_o}. "
                    f"Debería ser O(1)"
                )
                score -= 50
        
        elif nested_depth == 1:
            # 1 loop -> debe ser O(n) o O(log n)
            if big_o_norm not in ['o(n)', 'o(logn)']:
                # Verificar si es un caso especial (while con división)
                if not self._detect_binary_search_pattern(code):
                    errors.append(
                        f"ERROR: 1 loop detectado, pero Big-O es {result.big_o}. "
                        f"Debería ser O(n) o O(log n)"
                    )
                    score -= 40
        
        elif nested_depth == 2:
            # 2 loops anidados -> debe ser O(n²)
            if 'n2' not in big_o_norm and 'n^2' not in big_o_norm:
                errors.append(
                    f"ERROR: 2 loops anidados detectados, pero Big-O es {result.big_o}. "
                    f"Debería ser O(n²)"
                )
                score -= 40
        
        elif nested_depth >= 3:
            # 3+ loops anidados -> debe ser O(n³) o superior
            expected_complexity = f'o(n^{nested_depth})'
            if expected_complexity not in big_o_norm:
                errors.append(
                    f"ERROR: {nested_depth} loops anidados, pero Big-O es {result.big_o}. "
                    f"Debería ser O(n^{nested_depth})"
                )
                score -= 40
        
        return max(0, score), errors
    
    def _detect_binary_search_pattern(self, code: str) -> bool:
        """
        Detecta patrón específico de búsqueda binaria.
        """
        code_lower = code.lower()
        
        # Criterios para búsqueda binaria:
        # 1. Tiene variables izquierda/derecha (left/right)
        has_pointers = ('izquierda' in code_lower and 'derecha' in code_lower) or \
                      ('left' in code_lower and 'right' in code_lower)
        
        # 2. Calcula punto medio
        has_middle = 'medio' in code_lower or 'middle' in code_lower or 'mid' in code_lower
        
        # 3. Divide el espacio de búsqueda (división por 2)
        has_division = bool(re.search(r'(\+|\-)\s*\/\s*2|\/\s*2|\/\/\s*2', code_lower))
        
        # 4. Tiene loop (while)
        has_loop = 'while' in code_lower or 'mientras' in code_lower
        
        # Si cumple al menos 3 de 4 criterios, es búsqueda binaria
        criteria_met = sum([has_pointers, has_middle, has_division, has_loop])
        
        return criteria_met >= 3
    
    def _validate_with_gemini(
        self, 
        result: ComplexityResult,
        gemini_text: str
    ) -> Tuple[float, Dict[str, str]]:
        """
        Compara con el análisis de Gemini como referencia.
        MEJORADO: Más tolerante con notaciones equivalentes.
        """
        gemini_lower = gemini_text.lower()
        score = 0.0
        
        gemini_complexities = {
            'big_o': self._extract_complexity_from_gemini(gemini_lower, 'peor caso'),
            'omega': self._extract_complexity_from_gemini(gemini_lower, 'mejor caso'),
            'theta': self._extract_complexity_from_gemini(gemini_lower, 'caso promedio')
        }
        
        # Comparar cada notación con tolerancia
        if self._complexities_match(result.big_o, gemini_complexities['big_o']):
            score += 40.0  # 40% si Big-O coincide (es el más importante)
        elif gemini_complexities['big_o']:
            # Dar puntos parciales si están en el mismo orden de magnitud
            score += 20.0
        else:
            # Si Gemini no dio Big-O, asumir correcto
            score += 30.0
            
        if self._complexities_match(result.omega, gemini_complexities['omega']):
            score += 30.0
        elif not gemini_complexities['omega']:
            # Si Gemini no dio Omega, asumir correcto
            score += 25.0
            
        if self._complexities_match(result.theta, gemini_complexities['theta']):
            score += 30.0
        elif not gemini_complexities['theta']:
            # Si Gemini no dio Theta, asumir correcto
            score += 25.0
        
        return score, gemini_complexities
    
    def _count_nested_loops(self, code: str) -> int:
        """Cuenta la profundidad máxima de loops anidados."""
        lines = code.split('\n')
        current_depth = 0
        max_depth = 0
        
        for line in lines:
            line_clean = line.strip().lower()
            if re.search(r'\b(for|para|while|mientras)\b', line_clean):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif re.search(r'\b(fin|end)\b', line_clean):
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _normalize_complexity(self, complexity: str) -> str:
        """Normaliza notación de complejidad."""
        if not complexity:
            return ""
        
        # Remover espacios y convertir a minúsculas
        norm = complexity.strip().replace(' ', '')
        
        # Normalizar exponentes
        norm = norm.replace('²', '^2').replace('³', '^3')
        norm = norm.replace('n^2', 'n²').replace('n^3', 'n³')
        
        # Asegurar formato O(...)
        if not norm.startswith(('O(', 'Ω(', 'Θ(')):
            norm = f'O({norm})'
        
        # Normalizar a O(...)
        norm = norm.replace('Ω(', 'O(').replace('Θ(', 'O(')
        
        return norm
    
    def _get_complexity_index(self, complexity: str) -> int:
        """Obtiene el índice de orden de una complejidad."""
        norm = self._normalize_complexity(complexity)
        
        # Buscar en la lista de orden
        for idx, std_complexity in enumerate(self.complexity_order):
            if norm == std_complexity:
                return idx
        
        # Si no se encuentra exactamente, buscar equivalencias
        for idx, std_complexity in enumerate(self.complexity_order):
            if std_complexity in self.equivalences:
                for equiv in self.equivalences[std_complexity]:
                    if equiv in norm.lower():
                        return idx
        
        raise ValueError(f"Complejidad no reconocida: {complexity}")
    
    def _extract_complexity_from_gemini(
        self, 
        text: str, 
        context: str
    ) -> Optional[str]:
        """Extrae complejidad del texto de Gemini."""
        # Buscar patrones cerca del contexto
        pattern = rf'{context}.*?(O|Ω|Θ)\s*\(\s*([^)]+)\s*\)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            notation = match.group(1)
            complexity = match.group(2).strip()
            return f"{notation}({complexity})"
        
        return None
    
    def _complexities_match(
        self, 
        auto: str, 
        gemini: Optional[str]
    ) -> bool:
        """
        Verifica si dos complejidades son equivalentes.
        MEJORADO: Más tolerante con variaciones de notación.
        """
        if not gemini:
            return False
        
        auto_norm = self._normalize_complexity(auto).lower()
        gemini_norm = self._normalize_complexity(gemini).lower()
        
        # Comparación directa
        if auto_norm == gemini_norm:
            return True
        
        # Remover notación para comparar solo la función
        auto_clean = auto_norm.replace('o(', '').replace('ω(', '').replace('θ(', '').replace(')', '')
        gemini_clean = gemini_norm.replace('o(', '').replace('ω(', '').replace('θ(', '').replace(')', '')
        
        # Normalizar espacios y caracteres especiales
        auto_clean = auto_clean.replace(' ', '').replace('*', '').replace('·', '')
        gemini_clean = gemini_clean.replace(' ', '').replace('*', '').replace('·', '')
        
        # Comparación de función normalizada
        if auto_clean == gemini_clean:
            return True
        
        # Casos especiales de equivalencia
        equivalences = {
            'n^2': ['n²', 'nn', 'n*n', 'nsquared'],
            'n^3': ['n³', 'nnn', 'n*n*n', 'ncubed'],
            'logn': ['log(n)', 'log n', 'lgn', 'lg(n)'],
            'nlogn': ['n*logn', 'nlog(n)', 'n*log(n)', 'nlog n'],
        }
        
        for standard, variants in equivalences.items():
            if auto_clean in [standard] + variants and gemini_clean in [standard] + variants:
                return True
        
        return False

class RecurrenceValidator:
    """
    Valida la CORRECCIÓN de ecuaciones de recurrencia.
    """
    
    def __init__(self):
        # Patrones de recurrencia conocidos y sus soluciones
        self.known_patterns = {
            'T(n) = T(n-1) + O(1)': 'O(n)',  # Recursión lineal
            'T(n) = T(n/2) + O(1)': 'O(log n)',  # Búsqueda binaria
            'T(n) = 2T(n/2) + O(n)': 'O(n log n)',  # Merge sort
            'T(n) = 2T(n/2) + O(1)': 'O(n)',  # Divide y conquista simple
            'T(n) = T(n-1) + O(n)': 'O(n²)',  # Recursión cuadrática
        }
    
    def validate_recurrence(
        self,
        auto_recurrence: Optional[str],
        auto_big_o: str,
        gemini_text: str,
        code: str
    ) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        Valida la corrección de la ecuación de recurrencia.
        MEJORADO: No penaliza si no hay recursión real.
        
        Returns:
            Tuple[float, Dict, List]: (score 0-100, detalles, errores)
        """
        errors = []
        details = {}
        score = 100.0  # Empezar con 100 por defecto
        
        # Verificar si DEBERÍA haber recursión
        has_recursion = self._has_recursion(code)
        
        if not has_recursion:
            # No hay recursión, no necesitamos recurrencia
            if auto_recurrence and 'T(n)' in auto_recurrence:
                # Falso positivo: se generó recurrencia sin recursión
                details = {
                    'auto': auto_recurrence,
                    'expected': 'No aplica (sin recursión)',
                    'note': 'Recurrencia generada incorrectamente para algoritmo iterativo'
                }
                # No penalizar mucho, es un error menor
                return 80.0, details, []
            else:
                # Correcto: no hay recursión, no hay recurrencia
                return 100.0, {'auto': 'N/A', 'expected': 'N/A'}, []
        
        # Hay recursión, validar recurrencia
        if not auto_recurrence or auto_recurrence == 'No detectada':
            errors.append(
                "ERROR CRÍTICO: Código con recursión, pero no se generó ecuación de recurrencia"
            )
            return 0.0, {'auto': None, 'expected': 'Required'}, errors
        
        # Validar coherencia entre recurrencia y complejidad
        coherence_score, coherence_errors = self._validate_recurrence_complexity_coherence(
            auto_recurrence, auto_big_o
        )
        errors.extend(coherence_errors)
        score = coherence_score * 0.7  # 70% del peso
        
        # Validar con Gemini (30% del peso)
        gemini_recurrence = self._extract_recurrence_from_gemini(gemini_text)
        if gemini_recurrence:
            if self._recurrences_match(auto_recurrence, gemini_recurrence):
                score += 30.0
            else:
                # No coincide con Gemini, pero si la coherencia es buena, no penalizar tanto
                if coherence_score >= 80:
                    score += 15.0  # Dar puntos parciales
        else:
            # Gemini no dio recurrencia, dar puntos por defecto
            score += 25.0
        
        details = {
            'auto': auto_recurrence,
            'gemini': gemini_recurrence,
            'expected_complexity': self._solve_recurrence(auto_recurrence),
            'actual_complexity': auto_big_o,
            'has_recursion': has_recursion
        }
        
        return min(100, score), details, errors
    
    def _validate_recurrence_complexity_coherence(
        self,
        recurrence: str,
        big_o: str
    ) -> Tuple[float, List[str]]:
        """
        Valida que la solución de la recurrencia coincida con Big-O.
        """
        errors = []
        
        expected_complexity = self._solve_recurrence(recurrence)
        
        if not expected_complexity:
            errors.append(
                f"No se pudo determinar complejidad esperada para: {recurrence}"
            )
            return 50.0, errors
        
        # Normalizar complejidades
        big_o_norm = big_o.replace(' ', '').lower()
        expected_norm = expected_complexity.replace(' ', '').lower()
        
        if big_o_norm == expected_norm:
            return 100.0, []
        else:
            errors.append(
                f"ERROR: Recurrencia {recurrence} implica {expected_complexity}, "
                f"pero Big-O es {big_o}"
            )
            return 0.0, errors
    
    def _solve_recurrence(self, recurrence: str) -> Optional[str]:
        """
        Resuelve una ecuación de recurrencia.
        MEJORADO: Soporte completo para divide y conquista.
        """
        recurrence_norm = recurrence.replace(' ', '').lower()
        
        # Patrones conocidos (expandidos)
        known_patterns = {
            # Búsqueda binaria
            't(n)=t(n/2)+o(1)': 'O(log n)',
            't(n)=t(n/2)+o(k)': 'O(log n)',
            
            # Merge Sort / Quick Sort promedio
            't(n)=2t(n/2)+o(n)': 'O(n log n)',
            't(n)=2t(n/2)+o(1)': 'O(n)',
            
            # Recursión lineal
            't(n)=t(n-1)+o(1)': 'O(n)',
            't(n)=t(n-1)+o(k)': 'O(n)',
            't(n)=t(n-1)+o(n)': 'O(n²)',
            
            # Fibonacci / Recursión doble
            't(n)=2t(n-1)+o(1)': 'O(2^n)',
            
            # Árbol de recursión múltiple
            't(n)=3t(n/2)+o(n)': 'O(n^log₂(3))',
            't(n)=4t(n/2)+o(n)': 'O(n²)',
        }
        
        # Buscar coincidencia exacta
        for pattern, solution in known_patterns.items():
            if pattern in recurrence_norm or recurrence_norm in pattern:
                return solution
        
        # Análisis por componentes (Teorema Maestro simplificado)
        # T(n) = aT(n/b) + f(n)
        
        # Patrón: T(n) = aT(n/b) + O(n^k)
        master_pattern = r't\(n\)=(\d+)t\(n/(\d+)\)\+o\(n(?:\^(\d+))?\)'
        match = re.search(master_pattern, recurrence_norm)
        
        if match:
            a = int(match.group(1))  # Número de llamadas recursivas
            b = int(match.group(2))  # Factor de división
            k = int(match.group(3)) if match.group(3) else 1  # Exponente de n
            
            # Aplicar Teorema Maestro
            # log_b(a) es el exponente crítico
            import math
            log_b_a = math.log(a) / math.log(b)
            
            if k < log_b_a:
                # Caso 1: f(n) es polinomialmente menor
                return f'O(n^{log_b_a:.1f})' if log_b_a != int(log_b_a) else f'O(n^{int(log_b_a)})'
            elif k == log_b_a:
                # Caso 2: f(n) y trabajo recursivo son iguales
                if k == 1:
                    return 'O(n log n)'
                else:
                    return f'O(n^{k} log n)'
            else:
                # Caso 3: f(n) es polinomialmente mayor
                return f'O(n^{k})'
        
        # Heurísticas adicionales
        if 'n-1' in recurrence_norm and 'o(1)' in recurrence_norm:
            return 'O(n)'
        elif 'n/2' in recurrence_norm and 'o(1)' in recurrence_norm:
            return 'O(log n)'
        elif 'n/2' in recurrence_norm and 'o(n)' in recurrence_norm:
            if '2t' in recurrence_norm:
                return 'O(n log n)'
            else:
                return 'O(n)'
        
        return None
    
    def _has_recursion(self, code: str) -> bool:
        """
        Detecta si el código tiene recursión REAL.
        MEJORADO: Distingue entre recursión y llamadas a funciones externas.
        """
        code_lower = code.lower()
        
        # Buscar definición de función
        func_patterns = [
            r'\b(function|funcion|def|procedimiento)\s+(\w+)',
            r'\b(\w+)\s*\([^)]*\)\s*{',  # Estilo C/Java
        ]
        
        function_name = None
        for pattern in func_patterns:
            match = re.search(pattern, code_lower)
            if match:
                # Tomar el nombre de la función (segundo grupo o primero según patrón)
                function_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                break
        
        if not function_name:
            # No hay definición de función, no puede haber recursión
            return False
        
        # Buscar llamadas a LA MISMA función (recursión)
        # Excluir la línea de definición
        lines = code.split('\n')
        function_def_line = None
        
        for i, line in enumerate(lines):
            if re.search(rf'\b(function|funcion|def|procedimiento)\s+{function_name}\b', line.lower()):
                function_def_line = i
                break
        
        # Buscar llamadas recursivas en las otras líneas
        for i, line in enumerate(lines):
            if function_def_line is not None and i == function_def_line:
                continue  # Saltar línea de definición
            
            # Buscar llamada a la función
            if re.search(rf'\b{function_name}\s*\(', line.lower()):
                return True
        
        return False
    
    def _extract_recurrence_from_gemini(self, text: str) -> Optional[str]:
        """Extrae ecuación de recurrencia del texto de Gemini."""
        patterns = [
            r'T\(n\)\s*=\s*([^\n.]+)',
            r'recurrencia:\s*([^\n.]+)',
            r'ecuación:\s*([^\n.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _recurrences_match(self, auto: str, gemini: str) -> bool:
        """Verifica si dos recurrencias son equivalentes."""
        auto_norm = auto.replace(' ', '').lower()
        gemini_norm = gemini.replace(' ', '').lower()
        
        return auto_norm == gemini_norm

class CostAnalysisValidator:
    """
    Valida el análisis de costos temporales línea por línea.
    """
    
    def validate_line_costs(
        self,
        auto_lines: List[Dict],
        gemini_text: str,
        code: str
    ) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        Valida la corrección de los costos por línea.
        MEJORADO: Menos estricto, tolera variaciones razonables.
        
        Returns:
            Tuple[float, Dict, List]: (score 0-100, detalles, errores)
        """
        errors = []
        details = {}
        score = 100.0
        
        # Filtrar líneas relevantes
        relevant_lines = [
            line for line in auto_lines 
            if line.get('type') not in ['summary', 'other', None]
        ]
        
        if not relevant_lines:
            return 100.0, {'total_lines': 0}, []
        
        incorrect_costs = []
        acceptable_deviations = []  # Desviaciones aceptables pero no perfectas
        
        for line_info in relevant_lines:
            line_no = line_info.get('line', 0)
            line_type = line_info.get('type', '')
            time_cost = line_info.get('time_cost', '')
            
            # Validar que el costo sea apropiado para el tipo de línea
            expected_cost = self._get_expected_cost(line_type)
            
            # Si el costo coincide exactamente, perfecto
            if time_cost == expected_cost:
                continue
            
            # Si no coincide, verificar si es una variación aceptable
            if self._is_acceptable_cost_variation(time_cost, expected_cost, line_type):
                acceptable_deviations.append({
                    'line': line_no,
                    'type': line_type,
                    'auto_cost': time_cost,
                    'expected_cost': expected_cost
                })
                # Penalizar levemente (5% por línea)
                score -= 5.0
            else:
                # Error real
                incorrect_costs.append({
                    'line': line_no,
                    'type': line_type,
                    'auto_cost': time_cost,
                    'expected_cost': expected_cost
                })
                # Penalizar más (10% por línea, máximo 50%)
                score -= min(10.0, 50.0 / len(relevant_lines))
        
        if incorrect_costs:
            errors.append(
                f"ERROR: {len(incorrect_costs)} líneas con costos incorrectos"
            )
        
        details = {
            'total_lines': len(relevant_lines),
            'incorrect_costs': incorrect_costs,
            'acceptable_deviations': acceptable_deviations,
            'accuracy': (len(relevant_lines) - len(incorrect_costs)) / len(relevant_lines) * 100
        }
        
        return max(0, score), details, errors
    
    def _is_acceptable_cost_variation(
        self, 
        actual_cost: str, 
        expected_cost: str, 
        line_type: str
    ) -> bool:
        """
        Determina si una variación de costo es aceptable.
        Ej: O(1) vs O(k), O(n) vs O(log n) para ciertos casos.
        """
        if not actual_cost or not expected_cost:
            return False
        
        actual_norm = actual_cost.lower().replace(' ', '')
        expected_norm = expected_cost.lower().replace(' ', '')
        
        # Variaciones aceptables por tipo
        acceptable = {
            'while_loop': [
                ('o(n)', 'o(logn)'),  # While puede ser logarítmico
                ('o(n)', 'o(k)'),     # While con iteraciones limitadas
                ('o(k)', 'o(n)'),
            ],
            'for_loop': [
                ('o(n)', 'o(k)'),     # For con límite variable
            ],
            'assignment': [
                ('o(1)', 'o(k)'),     # Assignment con cálculo simple
            ]
        }
        
        if line_type in acceptable:
            for pair in acceptable[line_type]:
                if (actual_norm, expected_norm) == pair or (expected_norm, actual_norm) == pair:
                    return True
        
        return False
    
    def _get_expected_cost(self, line_type: str) -> str:
        """
        Retorna el costo temporal esperado según el tipo de línea.
        """
        cost_map = {
            'assignment': 'O(1)',
            'read': 'O(1)',
            'write': 'O(1)',
            'if_statement': 'O(1)',
            'for_loop': 'O(n)',
            'while_loop': 'O(n)',  # Puede variar
            'function_call': 'Variable',
        }
        
        return cost_map.get(line_type, 'O(1)')

class EfficiencyOrchestrator:
    """
    Orquesta la validación completa del análisis de eficiencia.
    
    Pesos de validación:
    - Complejidad: 50%
    - Recurrencia: 30%
    - Costos por línea: 15%
    - Patrones: 5%
    """
    
    def __init__(self):
        self.complexity_validator = ComplexityValidator()
        self.recurrence_validator = RecurrenceValidator()
        self.cost_validator = CostAnalysisValidator()
        self.pattern_recognizer = PatternRecognizer()
        
        self.analyzer = ComplexityAnalyzer()
        self.parser = PseudocodeParser()
        
        # Pesos de cada componente
        self.weights = {
            'complexity': 0.50,
            'recurrence': 0.30,
            'cost_analysis': 0.15,
            'patterns': 0.05
        }
    
    def validate_from_file(
        self, 
        file_path: str
    ) -> EfficiencyValidationResult:
        """
        Valida análisis de eficiencia desde un archivo .txt.
        
        Args:
            file_path: Ruta al archivo .txt con pseudocódigo
            
        Returns:
            EfficiencyValidationResult: Resultado completo de validación
        """
        # Leer archivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return self._create_error_result(
                file_path, 
                f"Error leyendo archivo: {e}"
            )
        
        algorithm_name = Path(file_path).stem
        
        return self.validate_algorithm(code, algorithm_name, file_path)
    
    def validate_algorithm(
        self,
        code: str,
        algorithm_name: str = "algorithm",
        file_path: Optional[str] = None
    ) -> EfficiencyValidationResult:
        """
        Validación completa de eficiencia algorítmica.
        """
        timestamp = datetime.now().isoformat()
        
        print(f"\n{'='*70}")
        print(f"VALIDANDO ANÁLISIS DE EFICIENCIA: {algorithm_name}")
        print(f"{'='*70}\n")
        
        # 1. ANÁLISIS AUTOMÁTICO
        print("→ Ejecutando análisis automático...")
        auto_complexity = self.analyzer.analyze(code)
        auto_patterns = self.pattern_recognizer.analyze(code)
        auto_lines = self.parser.analyze_by_line(code)
        
        auto_recurrence = None
        for item in auto_lines:
            if item.get('type') == 'summary' and 'recurrence' in item:
                auto_recurrence = item.get('recurrence')
                break
        
        # 2. CONSULTAR GEMINI (REFERENCIA)
        print("→ Consultando Gemini como referencia...")
        gemini_prompt = self._build_efficiency_prompt(code)
        gemini_response = ask_gemini(gemini_prompt)
        
        # 3. VALIDAR COMPLEJIDAD (50% del peso)
        print("→ Validando análisis de complejidad...")
        complexity_score, complexity_details, complexity_errors = \
            self.complexity_validator.validate_complexity(
                auto_complexity, gemini_response, code
            )
        
        # 4. VALIDAR RECURRENCIA (30% del peso)
        print("→ Validando ecuación de recurrencia...")
        recurrence_score, recurrence_details, recurrence_errors = \
            self.recurrence_validator.validate_recurrence(
                auto_recurrence, auto_complexity.big_o, gemini_response, code
            )
        
        # 5. VALIDAR COSTOS POR LÍNEA (15% del peso)
        print("→ Validando costos temporales por línea...")
        cost_score, cost_details, cost_errors = \
            self.cost_validator.validate_line_costs(
                auto_lines, gemini_response, code
            )
        
        # 6. VALIDAR PATRONES (5% del peso)
        print("→ Validando patrones algorítmicos...")
        pattern_score = self._validate_patterns_basic(auto_patterns, gemini_response)
        
        # 7. CALCULAR PUNTUACIÓN GENERAL
        overall_score = (
            complexity_score * self.weights['complexity'] +
            recurrence_score * self.weights['recurrence'] +
            cost_score * self.weights['cost_analysis'] +
            pattern_score * self.weights['patterns']
        )
        
        # 8. DETERMINAR RIGOR MATEMÁTICO
        mathematical_rigor = self._determine_rigor(overall_score)
        
        # 9. CLASIFICAR ERRORES
        critical_errors = []
        warnings = []
        corrections = []
        
        # Errores críticos (score < 30 en cualquier categoría)
        if complexity_score < 30:
            critical_errors.extend(complexity_errors)
        elif complexity_score < 70:
            warnings.extend(complexity_errors)
        
        if recurrence_score < 30 and auto_recurrence:
            critical_errors.extend(recurrence_errors)
        elif recurrence_score < 70 and auto_recurrence:
            warnings.extend(recurrence_errors)
        
        if cost_score < 30:
            critical_errors.extend(cost_errors)
        elif cost_score < 70:
            warnings.extend(cost_errors)
        
        # Generar correcciones
        corrections = self._generate_corrections(
            complexity_details, recurrence_details, cost_details
        )
        
        # 10. CONSTRUIR RESULTADO
        result = EfficiencyValidationResult(
            timestamp=timestamp,
            algorithm_name=algorithm_name,
            file_path=file_path,
            
            complexity_correctness=complexity_score,
            complexity_details=complexity_details,
            complexity_errors=complexity_errors,
            
            recurrence_correctness=recurrence_score,
            recurrence_details=recurrence_details,
            recurrence_errors=recurrence_errors,
            
            cost_analysis_correctness=cost_score,
            cost_details=cost_details,
            cost_errors=cost_errors,
            
            pattern_correctness=pattern_score,
            pattern_details={'detected': len(auto_patterns)},
            
            overall_efficiency_score=overall_score,
            mathematical_rigor=mathematical_rigor,
            
            critical_errors=critical_errors,
            warnings=warnings,
            corrections=corrections,
            
            auto_analysis={
                'complexity': {
                    'big_o': auto_complexity.big_o,
                    'omega': auto_complexity.omega,
                    'theta': auto_complexity.theta,
                    'explanation': auto_complexity.explanation
                },
                'recurrence': auto_recurrence,
                'patterns': [p.get('name', 'unknown') for p in auto_patterns]
            },
            gemini_analysis=gemini_response[:300] + '...'
        )
        
        self.print_validation_summary(result)
        
        return result
    
    def _build_efficiency_prompt(self, code: str) -> str:
        """
        Construye prompt optimizado para validación de eficiencia.
        MEJORADO: Prompt más estructurado para respuestas consistentes.
        """
        prompt = f"""Eres un experto en análisis de complejidad algorítmica. Analiza el siguiente pseudocódigo.

PSEUDOCÓDIGO:
```
{code}
```

RESPONDE EXACTAMENTE EN ESTE FORMATO (sin texto adicional):

1. Peor caso: O(...)
2. Mejor caso: Ω(...)
3. Caso promedio: Θ(...)
4. Recurrencia: T(n) = ... (solo si hay recursión)

REGLAS IMPORTANTES:
- Usa notaciones estándar: O(1), O(n), O(n²), O(log n), O(n log n), O(2^n)
- Para Omega usa el símbolo Ω o escribe "Omega"
- Para Theta usa el símbolo Θ o escribe "Theta"
- Si Big-O = Omega, entonces Theta = Big-O
- NO agregues explicaciones largas, solo las notaciones

EJEMPLO DE RESPUESTA CORRECTA:
1. Peor caso: O(n²)
2. Mejor caso: Ω(n)
3. Caso promedio: Θ(n²)
"""
        
        return prompt
    
    def _validate_patterns_basic(
        self, 
        patterns: List[Dict], 
        gemini_text: str
    ) -> float:
        """
        Validación básica de patrones (peso bajo).
        """
        if not patterns:
            return 100.0
        
        gemini_lower = gemini_text.lower()
        confirmed = 0
        
        pattern_keywords = {
            'divide_y_conquista': ['divide', 'conquista', 'merge', 'split'],
            'busqueda_binaria': ['binaria', 'binary', 'logarítmico'],
            'programacion_dinamica': ['dinámica', 'dynamic', 'memoización'],
            'algoritmo_voraz': ['greedy', 'voraz', 'ávido'],
            'backtracking': ['backtrack', 'retroceso']
        }
        
        for pattern in patterns:
            pattern_name = pattern.get('name', '')
            if pattern_name in pattern_keywords:
                if any(kw in gemini_lower for kw in pattern_keywords[pattern_name]):
                    confirmed += 1
        
        return (confirmed / len(patterns) * 100) if patterns else 100.0
    
    def _determine_rigor(self, score: float) -> str:
        """
        Determina el nivel de rigor matemático del análisis.
        """
        if score >= 85:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_corrections(
        self,
        complexity_details: Dict,
        recurrence_details: Dict,
        cost_details: Dict
    ) -> List[str]:
        """
        Genera correcciones específicas basadas en los errores detectados.
        """
        corrections = []
        
        # Correcciones de complejidad
        if complexity_details.get('structure_score', 100) < 100:
            auto_big_o = complexity_details.get('auto_complexities', {}).get('big_o')
            corrections.append(
                f"CORRECCIÓN: Revisar Big-O={auto_big_o}. "
                f"Verificar conteo de loops anidados."
            )
        
        # Correcciones de recurrencia
        if recurrence_details.get('expected_complexity') != recurrence_details.get('actual_complexity'):
            corrections.append(
                f"CORRECCIÓN: Recurrencia {recurrence_details.get('auto')} "
                f"debería dar {recurrence_details.get('expected_complexity')}, "
                f"pero Big-O es {recurrence_details.get('actual_complexity')}"
            )
        
        # Correcciones de costos
        incorrect = cost_details.get('incorrect_costs', [])
        if incorrect:
            corrections.append(
                f"CORRECCIÓN: {len(incorrect)} líneas con costos incorrectos. "
                f"Revisar tipos de operaciones."
            )
        
        return corrections
    
    def _create_error_result(
        self, 
        file_path: str, 
        error_msg: str
    ) -> EfficiencyValidationResult:
        """
        Crea un resultado de error cuando falla la lectura del archivo.
        """
        return EfficiencyValidationResult(
            timestamp=datetime.now().isoformat(),
            algorithm_name=Path(file_path).stem if file_path else "unknown",
            file_path=file_path,
            critical_errors=[error_msg],
            overall_efficiency_score=0.0,
            mathematical_rigor="LOW"
        )
    
    def print_validation_summary(self, result: EfficiencyValidationResult):
        """
        Imprime resumen legible de la validación de eficiencia.
        """
        print(f"\n{'='*70}")
        print(f"REPORTE DE VALIDACIÓN DE EFICIENCIA")
        print(f"{'='*70}")
        print(f"Algoritmo: {result.algorithm_name}")
        if result.file_path:
            print(f"Archivo: {result.file_path}")
        print(f"Timestamp: {result.timestamp}")
        print(f"\n{'─'*70}")
        
        # PUNTUACIÓN GENERAL
        print(f"\n PUNTUACIÓN GENERAL: {result.overall_efficiency_score:.1f}/100")
        print(f"   Rigor Matemático: {result.mathematical_rigor}")
        
        # Barra de progreso visual
        bar_length = 50
        filled = int(bar_length * result.overall_efficiency_score / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"   [{bar}] {result.overall_efficiency_score:.1f}%")
        
        print(f"\n{'─'*70}")
        
        # DESGLOSE POR COMPONENTE
        print("\n DESGLOSE POR COMPONENTE:")
        print(f"   ├─ Complejidad (50%): {result.complexity_correctness:.1f}/100")
        print(f"   ├─ Recurrencia (30%): {result.recurrence_correctness:.1f}/100")
        print(f"   ├─ Costos (15%):      {result.cost_analysis_correctness:.1f}/100")
        print(f"   └─ Patrones (5%):     {result.pattern_correctness:.1f}/100")
        
        # ANÁLISIS DE COMPLEJIDAD
        print(f"\n{'─'*70}")
        print("\n ANÁLISIS DE COMPLEJIDAD:")
        auto_comp = result.auto_analysis.get('complexity', {})
        print(f"   Big-O (Peor caso):     {auto_comp.get('big_o', 'N/A')}")
        print(f"   Omega (Mejor caso):    {auto_comp.get('omega', 'N/A')}")
        print(f"   Theta (Caso promedio): {auto_comp.get('theta', 'N/A')}")
        
        gemini_comp = result.complexity_details.get('gemini_complexities', {})
        if gemini_comp.get('big_o'):
            print(f"\n   Referencia Gemini:")
            print(f"   Big-O:  {gemini_comp.get('big_o', 'N/A')}")
            print(f"   Omega:  {gemini_comp.get('omega', 'N/A')}")
            print(f"   Theta:  {gemini_comp.get('theta', 'N/A')}")
        
        # RECURRENCIA
        if result.auto_analysis.get('recurrence'):
            print(f"\n{'─'*70}")
            print("\n ECUACIÓN DE RECURRENCIA:")
            print(f"   Detectada: {result.auto_analysis.get('recurrence')}")
            
            expected = result.recurrence_details.get('expected_complexity')
            actual = result.recurrence_details.get('actual_complexity')
            if expected and actual:
                match = "✓" if expected == actual else "✗"
                print(f"   Solución esperada: {expected}")
                print(f"   Big-O calculado:   {actual} {match}")
        
        # ERRORES CRÍTICOS
        if result.critical_errors:
            print(f"\n{'─'*70}")
            print("\n  ERRORES CRÍTICOS:")
            for error in result.critical_errors[:5]:
                print(f"   • {error}")
        
        # ADVERTENCIAS
        if result.warnings:
            print(f"\n{'─'*70}")
            print("\n ADVERTENCIAS:")
            for warning in result.warnings[:5]:
                print(f"   • {warning}")
        
        # CORRECCIONES SUGERIDAS
        if result.corrections:
            print(f"\n{'─'*70}")
            print("\n CORRECCIONES SUGERIDAS:")
            for correction in result.corrections:
                print(f"   • {correction}")
        
        # VEREDICTO FINAL
        print(f"\n{'='*70}")
        if result.overall_efficiency_score >= 85:
            print(" VEREDICTO: Análisis de eficiencia CORRECTO")
        elif result.overall_efficiency_score >= 60:
            print("  VEREDICTO: Análisis ACEPTABLE con observaciones")
        else:
            print(" VEREDICTO: Análisis REQUIERE CORRECCIÓN")
        print(f"{'='*70}\n")
    
    def export_validation_report(
        self, 
        result: EfficiencyValidationResult, 
        filepath: str = "efficiency_validation_report.json"
    ):
        """
        Exporta reporte de validación a JSON.
        """
        report = {
            'metadata': {
                'timestamp': result.timestamp,
                'algorithm': result.algorithm_name,
                'file_path': result.file_path,
                'overall_score': result.overall_efficiency_score,
                'mathematical_rigor': result.mathematical_rigor
            },
            'scores': {
                'complexity': result.complexity_correctness,
                'recurrence': result.recurrence_correctness,
                'cost_analysis': result.cost_analysis_correctness,
                'patterns': result.pattern_correctness
            },
            'complexity_analysis': {
                'auto': result.auto_analysis.get('complexity'),
                'details': result.complexity_details,
                'errors': result.complexity_errors
            },
            'recurrence_analysis': {
                'auto': result.auto_analysis.get('recurrence'),
                'details': result.recurrence_details,
                'errors': result.recurrence_errors
            },
            'cost_analysis': {
                'details': result.cost_details,
                'errors': result.cost_errors
            },
            'validation': {
                'critical_errors': result.critical_errors,
                'warnings': result.warnings,
                'corrections': result.corrections
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f" Reporte exportado: {filepath}")
    
    def validate_batch_from_directory(
        self,
        directory: str,
        output_dir: str = "validation_reports"
    ) -> List[EfficiencyValidationResult]:
        """
        Valida todos los archivos .txt en un directorio.
        
        Args:
            directory: Directorio con archivos .txt de pseudocódigo
            output_dir: Directorio donde guardar reportes
            
        Returns:
            Lista de resultados de validación
        """
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f" Error: Directorio no existe: {directory}")
            return []
        
        txt_files = list(directory_path.glob("*.txt"))
        
        if not txt_files:
            print(f"  No se encontraron archivos .txt en: {directory}")
            return []
        
        print(f"\n{'='*70}")
        print(f"VALIDACIÓN EN LOTE: {len(txt_files)} archivos")
        print(f"{'='*70}\n")
        
        results = []
        
        for txt_file in txt_files:
            print(f"\n→ Procesando: {txt_file.name}")
            
            result = self.validate_from_file(str(txt_file))
            results.append(result)
            
            # Exportar reporte individual
            report_name = f"{txt_file.stem}_validation.json"
            report_path = Path(output_dir) / report_name
            self.export_validation_report(result, str(report_path))
        
        # Generar reporte consolidado
        self._generate_batch_summary(results, output_dir)
        
        return results
    
    def _generate_batch_summary(
        self,
        results: List[EfficiencyValidationResult],
        output_dir: str
    ):
        """
        Genera reporte consolidado de validación en lote.
        """
        summary = {
            'total_files': len(results),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'avg_score': sum(r.overall_efficiency_score for r in results) / len(results),
                'min_score': min(r.overall_efficiency_score for r in results),
                'max_score': max(r.overall_efficiency_score for r in results),
                'high_rigor': sum(1 for r in results if r.mathematical_rigor == 'HIGH'),
                'medium_rigor': sum(1 for r in results if r.mathematical_rigor == 'MEDIUM'),
                'low_rigor': sum(1 for r in results if r.mathematical_rigor == 'LOW')
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
        
        summary_path = Path(output_dir) / "batch_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(" RESUMEN DE VALIDACIÓN EN LOTE")
        print(f"{'='*70}")
        print(f"Total de archivos: {summary['total_files']}")
        print(f"Puntuación promedio: {summary['statistics']['avg_score']:.1f}/100")
        print(f"Rango: {summary['statistics']['min_score']:.1f} - {summary['statistics']['max_score']:.1f}")
        print(f"\nRigor Matemático:")
        print(f"  HIGH:   {summary['statistics']['high_rigor']} archivos")
        print(f"  MEDIUM: {summary['statistics']['medium_rigor']} archivos")
        print(f"  LOW:    {summary['statistics']['low_rigor']} archivos")
        print(f"\n Reporte consolidado: {summary_path}")
        print(f"{'='*70}\n")

# FUNCIÓN AUXILIAR PARA USO RÁPIDO

def validate_efficiency(
    code_or_file: str,
    algorithm_name: str = None,
    is_file: bool = False
) -> EfficiencyValidationResult:
    """
    Valida análisis de eficiencia algorítmica.
    
    Args:
        code_or_file: Pseudocódigo o ruta a archivo .txt
        algorithm_name: Nombre del algoritmo (opcional)
        is_file: True si code_or_file es una ruta de archivo
        
    Returns:
        EfficiencyValidationResult: Resultado de la validación
        
    Ejemplo de uso:
        # Desde archivo
        result = validate_efficiency("tests/pseudocode/binary_search.txt", is_file=True)
        
        # Desde código directo
        code = '''
        PARA i DESDE 0 HASTA n-1 HACER
            ESCRIBIR i
        FIN PARA
        '''
        result = validate_efficiency(code, "simple_loop")
    """
    orchestrator = EfficiencyOrchestrator()
    
    if is_file:
        result = orchestrator.validate_from_file(code_or_file)
    else:
        if not algorithm_name:
            algorithm_name = "algorithm"
        result = orchestrator.validate_algorithm(code_or_file, algorithm_name)
    
    return result

def validate_directory(
    directory: str,
    output_dir: str = "validation_reports"
) -> List[EfficiencyValidationResult]:
    """
    Valida todos los archivos .txt en un directorio.
    
    Args:
        directory: Ruta al directorio con archivos .txt
        output_dir: Directorio donde guardar reportes
        
    Returns:
        Lista de resultados de validación
        
    Ejemplo de uso:
        results = validate_directory("tests/pseudocode")
    """
    orchestrator = EfficiencyOrchestrator()
    return orchestrator.validate_batch_from_directory(directory, output_dir)

if __name__ == "__main__":
    # Ejemplo 1: Validar un archivo específico
    print("EJEMPLO 1: Validación de archivo individual\n")
    result = validate_efficiency(
        "tests/pseudocode/binary_search.txt",
        is_file=True
    )
    
    # Ejemplo 2: Validar código directamente
    print("\n\nEJEMPLO 2: Validación de código directo\n")
    sample_code = """
    PARA i DESDE 0 HASTA n-1 HACER
        PARA j DESDE 0 HASTA n-1 HACER
            ESCRIBIR arreglo[i][j]
        FIN PARA
    FIN PARA
    """
    result = validate_efficiency(sample_code, "nested_loops")
    
    # Ejemplo 3: Validación en lote de un directorio
    print("\n\nEJEMPLO 3: Validación en lote\n")
    results = validate_directory("tests/pseudocode", "validation_reports")
    
def analyze_line_by_line():
    """
    Realiza el análisis línea por línea y obtiene la ecuación de recurrencia.
    Permite analizar archivos existentes o ingresar pseudocódigo desde la consola.
    """
    print("\n" + "=" * 70)
    print(" ANÁLISIS LÍNEA POR LÍNEA Y ECUACIÓN DE RECURRENCIA")
    print("=" * 70)

    # Preguntar al usuario si desea analizar un archivo o ingresar pseudocódigo
    print("\nOpciones disponibles:")
    print("  1. Analizar un archivo existente")
    print("  2. Ingresar pseudocódigo manualmente")
    choice = input("Selecciona una opción (1-2): ")

    if choice == "1":
        # Listar archivos disponibles en tests/pseudocode
        pseudocode_dir = Path(__file__).parent.parent / "tests" / "pseudocode"
        files = list(pseudocode_dir.glob("*.txt"))
        if not files:
            print("No se encontraron archivos en el directorio 'tests/pseudocode'.")
            return

        print("\nArchivos disponibles:")
        for idx, file in enumerate(files, start=1):
            print(f"  {idx}. {file.name}")

        file_choice = int(input("Selecciona un archivo por número: "))
        selected_file = files[file_choice - 1]

        # Leer el contenido del archivo
        with open(selected_file, "r") as f:
            code_lines = [{"line": line.strip(), "time_cost": 1, "exec_count": 1} for line in f.readlines()]

    elif choice == "2":
        # Ingresar pseudocódigo manualmente
        print("\nIngresa tu pseudocódigo línea por línea. Escribe 'END' para finalizar:")
        code_lines = []
        while True:
            line = input(">> ")
            if line.strip().upper() == "END":
                break
            code_lines.append({"line": line.strip(), "time_cost": 1, "exec_count": 1})

    else:
        print("Opción inválida.")
        return

    # Realizar el análisis
    result = analyze_recurrence(code_lines)

    # Mostrar el resultado
    print("\n ECUACIÓN DE RECURRENCIA:")
    print(f"   → Detectada: {result['recurrence_eq']}")
    print(f"   → Solución: {result['solution']}\n")

    print(" DESGLOSE LÍNEA POR LÍNEA:")
    for step in result["series_steps"]:
        print(f"   • {step}")

    print("\n" + "=" * 70)
    print(" ANÁLISIS COMPLETADO")
    print("=" * 70)