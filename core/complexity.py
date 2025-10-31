"""
Módulo `complexity.py`

Este módulo analiza la complejidad de algoritmos, incluyendo el cálculo de:
- Big-O (peor caso)
- Omega (mejor caso)
- Theta (caso promedio)

También detecta patrones algorítmicos como recursión, loops anidados y patrones logarítmicos.

Clases:
    - ComplexityResult: Representa el resultado del análisis de complejidad.
    - ComplexityAnalyzer: Clase principal para analizar la complejidad de algoritmos.

Funciones principales:
    - analyze: Realiza el análisis completo de un algoritmo.
    - _analyze_loops: Analiza loops simples y anidados.
    - _analyze_recursion: Detecta recursión y patrones relacionados.
    - _analyze_conditionals: Analiza condicionales y su impacto en la complejidad.
"""

from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass


@dataclass
class ComplexityResult:
    """
    Representa el resultado del análisis de complejidad.

    Atributos:
        big_o (str): Complejidad en el peor caso (Big-O).
        omega (str): Complejidad en el mejor caso (Omega).
        theta (str): Complejidad en el caso promedio (Theta).
        explanation (str): Explicación detallada del análisis.
        recurrence (Optional[str]): Fórmula de recurrencia (si aplica).
        pattern_type (Optional[str]): Tipo de patrón detectado (e.g., "divide y conquista").
    """
    big_o: str  # Peor caso
    omega: str  # Mejor caso
    theta: str  # Caso promedio
    explanation: str
    recurrence: Optional[str] = None
    pattern_type: Optional[str] = None


class ComplexityAnalyzer:
    """
    Clase para analizar la complejidad de algoritmos a partir de su pseudocódigo.

    Métodos principales:
        - analyze(code: str) -> ComplexityResult: Realiza el análisis completo de un algoritmo.
        - _analyze_loops() -> Dict[str, Any]: Analiza loops simples y anidados.
        - _analyze_recursion() -> Dict[str, Any]: Detecta recursión y patrones relacionados.
        - _analyze_conditionals() -> Dict[str, Any]: Analiza condicionales y su impacto en la complejidad.
        - _calculate_big_o(): Calcula la complejidad Big-O (peor caso).
        - _calculate_omega(): Calcula la complejidad Omega (mejor caso).
        - _calculate_theta(): Calcula la complejidad Theta (caso promedio).
    """
    
    def __init__(self) -> None:
        self.code: Optional[str] = None
        self.complexity_result: Optional[ComplexityResult] = None
        self.nested_depth: int = 0
        self.has_recursion: bool = False
        self.has_conditionals: bool = False

    def analyze(self, code: str) -> ComplexityResult:
        """
        Realiza el análisis completo de un algoritmo.

        Args:
            code (str): Pseudocódigo del algoritmo a analizar.

        Returns:
            ComplexityResult: Objeto con todas las complejidades (Big-O, Omega, Theta) y una explicación detallada.
        """
        self.code = code.lower()
        self.nested_depth = 0
        self.has_recursion = False
        self.has_conditionals = False

        # Análisis de estructura
        loop_complexity: Dict[str, Any] = self._analyze_loops()
        recursion_info: Dict[str, Any] = self._analyze_recursion()
        conditional_impact: Dict[str, Any] = self._analyze_conditionals()
        
        # Determinar complejidades
        big_o: str = self._calculate_big_o(loop_complexity, recursion_info)
        omega: str = self._calculate_omega(loop_complexity, recursion_info, conditional_impact)
        theta: str = self._calculate_theta(big_o, omega, conditional_impact)
        
        explanation: str = self._generate_explanation(
            loop_complexity, recursion_info, conditional_impact
        )
        
        self.complexity_result = ComplexityResult(
            big_o=big_o,
            omega=omega,
            theta=theta,
            explanation=explanation,
            recurrence=recursion_info.get('recurrence'),
            pattern_type=recursion_info.get('pattern')
        )
        
        return self.complexity_result

    def _analyze_loops(self) -> Dict[str, Any]:
        """
        Analiza loops simples y anidados en el pseudocódigo.

        Returns:
            Dict[str, Any]: Información sobre los loops detectados, incluyendo:
                - simple_loops (int): Número de loops simples.
                - nested_depth (int): Profundidad máxima de loops anidados.
                - while_loops (int): Número de loops `while`.
                - for_loops (int): Número de loops `for`.
                - has_logarithmic (bool): Indica si se detectaron patrones logarítmicos.
        """
        lines: List[str] = self.code.split('\n') if self.code else []
        loop_stack: List[Dict[str, Any]] = []  
        max_depth: int = 0
        
        loop_info: Dict[str, Any] = {
            'simple_loops': 0,
            'nested_depth': 0,
            'while_loops': 0,
            'for_loops': 0,
            'has_logarithmic': False
        }
        
        for line in lines:
            line_clean: str = line.strip()
            
            # Detectar inicio de loop
            if re.search(r'\b(for|para)\b.*\b(hacer|do)\b', line_clean, re.IGNORECASE):
                loop_type = 'for'
                loop_stack.append({'type': loop_type, 'line': line_clean})
                loop_info['for_loops'] += 1
                max_depth = max(max_depth, len(loop_stack))
                
            elif re.search(r'\b(while|mientras)\b.*\b(hacer|do)\b', line_clean, re.IGNORECASE):
                loop_type = 'while'
                loop_stack.append({'type': loop_type, 'line': line_clean})
                loop_info['while_loops'] += 1
                max_depth = max(max_depth, len(loop_stack))
                
                # Detectar patrón logarítmico (búsqueda binaria)
                if self._is_logarithmic_pattern(line_clean):
                    loop_info['has_logarithmic'] = True
            
            # Detectar fin de loop
            elif re.search(r'\b(fin|end)\b\s*(para|for|mientras|while)?\b', line_clean, re.IGNORECASE):
                if loop_stack:
                    loop_stack.pop()
        
        loop_info['nested_depth'] = max_depth
        loop_info['simple_loops'] = loop_info['for_loops'] + loop_info['while_loops']
        
        return loop_info

    def _is_logarithmic_pattern(self, line: str) -> Tuple[bool, float]:
        """Detecta patrones logarítmicos con análisis de confianza"""
        base_patterns = {
            # División o multiplicación por 2
            r'\/\s*2': 0.4,
            r'\*\s*2': 0.4,
            r'>>': 0.4,  # Desplazamiento binario
            r'<<': 0.4,
            
            # Búsqueda binaria
            r'izquierda.*derecha': 0.6,
            r'left.*right': 0.6,
            r'medio': 0.5,
            r'middle': 0.5,
            
            # Patrones específicos
            r'binary.*search': 0.8,
            r'búsqueda.*binaria': 0.8,
            r'log': 0.7,
            r'mitad': 0.5,
            r'half': 0.5
        }
        
        confidence = 0.0
        matches = []
        
        # Buscar coincidencias de patrones
        for pattern, score in base_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(pattern)
                confidence = max(confidence, score)
        
        # Análisis de combinaciones
        if 'medio' in line.lower() or 'middle' in line.lower():
            if 'izquierda' in line.lower() or 'left' in line.lower():
                if 'derecha' in line.lower() or 'right' in line.lower():
                    confidence = max(confidence, 0.9)  # Alta confianza para búsqueda binaria completa
        
        # Análisis de contexto
        if re.search(r'while|mientras', line, re.IGNORECASE):
            if any(re.search(pattern, line) for pattern in [r'\/\s*2', r'>>', r'<{']):
                confidence = min(1.0, confidence + 0.2)  # Bonus por while con división
        
        # Determinar si es logarítmico basado en umbral de confianza
        is_logarítmico = confidence >= 0.4
        
        return is_logarítmico, confidence

    def _analyze_recursion(self) -> Dict[str, Any]:
        """Analiza recursión con análisis semántico mejorado"""
        recursion_info: Dict[str, Any] = {
            'has_recursion': False,
            'pattern': None,
            'recurrence': None,
            'recursive_calls': 0,
            'pattern_confidence': 0.0,
            'has_base_case': False,
            'recursive_variables': set(),
            'operation_type': None
        }
        
        if not self.code:
            return recursion_info
        
        lines: List[str] = self.code.split('\n')
        function_name: Optional[str] = None
        
        # Buscar definición de función
        for line in lines:
            if re.search(r'\b(function|funcion|def|procedimiento)\b', line):
                match = re.search(r'\b(function|funcion|def|procedimiento)\s+(\w+)', line)
                if match:
                    function_name = match.group(2)
                    break
        
        if not function_name:
            return recursion_info
        
        # Contar llamadas recursivas
        recursive_calls: int = 0
        has_base_case: bool = False
        divides_problem: bool = False
        
        for line in lines:
            # Detectar llamadas recursivas
            if re.search(rf'\b{function_name}\s*\(', line):
                recursive_calls += 1
            
            # Detectar caso base
            if re.search(r'\breturn\b', line) and not re.search(rf'{function_name}', line):
                has_base_case = True
            
            # Detectar división del problema
            if re.search(r'\/\s*2|\/\/\s*2|\*\s*2', line):
                divides_problem = True
        
        if recursive_calls > 0:
            recursion_info['has_recursion'] = True
            recursion_info['recursive_calls'] = recursive_calls
            self.has_recursion = True
            
            # Clasificar patrón
            if recursive_calls == 1 and divides_problem:
                recursion_info['pattern'] = 'divide_y_conquista_simple'
                recursion_info['recurrence'] = 'T(n) = T(n/2) + O(1)'
            elif recursive_calls == 2 and divides_problem:
                recursion_info['pattern'] = 'divide_y_conquista_doble'
                recursion_info['recurrence'] = 'T(n) = 2T(n/2) + O(n)'
            elif recursive_calls == 1:
                recursion_info['pattern'] = 'recursion_lineal'
                recursion_info['recurrence'] = 'T(n) = T(n-1) + O(1)'
            else:
                recursion_info['pattern'] = 'recursion_multiple'
                recursion_info['recurrence'] = f'T(n) = {recursive_calls}T(n-1) + O(1)'
        
        return recursion_info

    def _analyze_conditionals(self) -> Dict[str, Any]:
        """Analiza condicionales y su impacto en complejidad"""
        conditional_info: Dict[str, Any] = {
            'count': 0,
            'nested': False,
            'affects_omega': False,
            'early_return': False
        }
        
        if not self.code:
            return conditional_info
        
        lines: List[str] = self.code.split('\n')
        if_depth: int = 0
        
        for line in lines:
            # Contar condicionales
            if re.search(r'\b(if|si)\b', line):
                if_depth += 1
                conditional_info['count'] += 1
                if if_depth > 1:
                    conditional_info['nested'] = True
                
                # Detectar early return (afecta mejor caso)
                if re.search(r'\breturn\b', line):
                    conditional_info['early_return'] = True
                    conditional_info['affects_omega'] = True
            
            elif re.search(r'\b(end|fin)\b', line):
                if if_depth > 0:
                    if_depth -= 1
        
        self.has_conditionals = conditional_info['count'] > 0
        return conditional_info

    def _calculate_big_o(self, loop_info: Dict[str, Any], recursion_info: Dict[str, Any]) -> str:
        """Calcula la complejidad O (peor caso)"""
        
        # Recursión tiene prioridad
        if recursion_info['has_recursion']:
            pattern: Optional[str] = recursion_info.get('pattern')
            if pattern == 'divide_y_conquista_simple':
                return "O(log n)"
            elif pattern == 'divide_y_conquista_doble':
                return "O(n log n)"
            elif pattern == 'recursion_lineal':
                return "O(n)"
            elif pattern == 'recursion_multiple':
                calls: int = recursion_info.get('recursive_calls', 2)
                return f"O({calls}^n)"
        
        # Loops logarítmicos
        if loop_info.get('has_logarithmic', False):
            return "O(log n)"
        
        # Loops anidados
        depth: int = loop_info.get('nested_depth', 0)
        if depth == 0:
            return "O(1)"
        elif depth == 1:
            return "O(n)"
        elif depth == 2:
            return "O(n²)"
        elif depth == 3:
            return "O(n³)"
        else:
            return f"O(n^{depth})"

    def _calculate_omega(self, loop_info: Dict[str, Any], recursion_info: Dict[str, Any], 
                        conditional_info: Dict[str, Any]) -> str:
        """Calcula la complejidad Ω (mejor caso)"""
        
        # Si hay early return, el mejor caso puede ser constante
        if conditional_info.get('early_return', False):
            return "Ω(1)"
        
        # Si hay recursión sin early return
        if recursion_info.get('has_recursion', False):
            # El mejor caso suele ser el caso base
            return "Ω(1)"
        
        # Para loops, el mejor caso es similar al peor caso
        # a menos que haya condicionales con break
        code_check: str = self.code if self.code else ""
        if 'break' in code_check or 'salir' in code_check:
            return "Ω(1)"
        
        # Sin optimizaciones, Ω es igual a O
        big_o: str = self._calculate_big_o(loop_info, recursion_info)
        return big_o.replace('O', 'Ω')

    def _calculate_theta(self, big_o: str, omega: str, 
                        conditional_info: Dict[str, Any]) -> str:
        """Calcula la complejidad Θ (caso promedio)"""
        
        # Si O y Ω son iguales, entonces Θ es el mismo
        big_o_value: str = big_o.replace('O', '')
        omega_value: str = omega.replace('Ω', '')
        
        if big_o_value == omega_value:
            return big_o.replace('O', 'Θ')
        
        # Si hay diferencia significativa entre mejor y peor caso
        # el caso promedio suele ser el peor caso
        if conditional_info.get('affects_omega', False):
            # Caso promedio es más cercano al peor caso
            return big_o.replace('O', 'Θ')
        
        return big_o.replace('O', 'Θ')

    def _generate_explanation(self, loop_info: Dict[str, Any], recursion_info: Dict[str, Any],
                            conditional_info: Dict[str, Any]) -> str:
        """Genera explicación detallada del análisis"""
        explanation: List[str] = []
        
        if recursion_info.get('has_recursion', False):
            pattern: Optional[str] = recursion_info.get('pattern')
            explanation.append(f"Algoritmo recursivo detectado ({pattern})")
            recurrence: Optional[str] = recursion_info.get('recurrence')
            if recurrence:
                explanation.append(f"Recurrencia: {recurrence}")
        
        nested_depth: int = loop_info.get('nested_depth', 0)
        if nested_depth == 1:
            explanation.append("Loop simple detectado → O(n)")
        elif nested_depth == 2:
            explanation.append("Loops anidados (2 niveles) detectados → O(n²)")
        elif nested_depth > 2:
            explanation.append(
                f"Loops anidados ({nested_depth} niveles) → O(n^{nested_depth})"
            )
        
        if loop_info.get('has_logarithmic', False):
            explanation.append("Patrón logarítmico detectado (búsqueda binaria o similar)")
        
        if conditional_info.get('affects_omega', False):
            explanation.append(
                "Condicionales con early return afectan Ω (mejor caso puede ser O(1))"
            )
        else:
            explanation.append(
                "Condicionales no cambian el orden de complejidad"
            )
        
        return " | ".join(explanation) if explanation else "Sin información de análisis"

    def get_complexity(self) -> str:
        """Retorna resumen de complejidades"""
        if not self.complexity_result:
            return "No analysis performed yet"
        
        result: ComplexityResult = self.complexity_result
        return (
            f"Big-O (peor caso): {result.big_o}\n"
            f"Omega (mejor caso): {result.omega}\n"
            f"Theta (caso promedio): {result.theta}\n"
            f"Explicación: {result.explanation}"
        )

    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Retorna análisis completo en formato diccionario"""
        if not self.complexity_result:
            return {"error": "No analysis performed yet"}
        
        result: ComplexityResult = self.complexity_result
        return {
            "big_o": result.big_o,
            "omega": result.omega,
            "theta": result.theta,
            "explanation": result.explanation,
            "recurrence": result.recurrence,
            "pattern_type": result.pattern_type
        }