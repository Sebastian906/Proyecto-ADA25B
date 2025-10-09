"""
Módulo para análisis de complejidad de algoritmos (O, Ω, Θ)
Incluye análisis detallado de loops, recursión y patrones algorítmicos
"""

from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass

@dataclass
class ComplexityResult:
    """Resultado del análisis de complejidad"""
    big_o: str  # Peor caso
    omega: str  # Mejor caso
    theta: str  # Caso promedio
    explanation: str
    recurrence: Optional[str] = None
    pattern_type: Optional[str] = None

class ComplexityAnalyzer:
    """
    Clase para analizar la complejidad de algoritmos a partir de su pseudocódigo
    Detecta O (peor caso), Ω (mejor caso), Θ (caso promedio)
    """
    def __init__(self):
        self.code = None
        self.complexity_result = None
        self.nested_depth = 0
        self.has_recursion = False
        self.has_conditionals = False

    def analyze(self, code: str) -> ComplexityResult:
        """
        Analiza la complejidad completa de un algoritmo
        
        Args:
            code (str): Pseudocódigo del algoritmo a analizar
            
        Returns:
            ComplexityResult: Objeto con todas las complejidades (O, Ω, Θ)
        """
        self.code = code.lower()
        self.nested_depth = 0
        self.has_recursion = False
        self.has_conditionals = False

        # Análisis de estructura
        loop_complexity = self._analyze_loops()
        recursion_info = self._analyze_recursion()
        conditional_impact = self._analyze_conditionals()
        
        # Determinar complejidades
        big_o = self._calculate_big_o(loop_complexity, recursion_info)
        omega = self._calculate_omega(loop_complexity, recursion_info, conditional_impact)
        theta = self._calculate_theta(big_o, omega, conditional_impact)
        
        explanation = self._generate_explanation(
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

    def _analyze_loops(self) -> Dict:
        """Analiza loops simples y anidados"""
        lines = self.code.split('\n')
        loop_stack = []
        max_depth = 0
        current_depth = 0
        loop_info = {
            'simple_loops': 0,
            'nested_depth': 0,
            'while_loops': 0,
            'for_loops': 0,
            'has_logarithmic': False
        }
        
        for line in lines:
            line = line.strip()
            
            # Detectar inicio de loop
            if re.search(r'\b(for|para)\b', line):
                current_depth += 1
                loop_stack.append('for')
                loop_info['for_loops'] += 1
                max_depth = max(max_depth, current_depth)
                
            elif re.search(r'\bwhile\b|\bmientras\b', line):
                current_depth += 1
                loop_stack.append('while')
                loop_info['while_loops'] += 1
                max_depth = max(max_depth, current_depth)
                
                # Detectar patrón logarítmico (búsqueda binaria)
                if self._is_logarithmic_pattern(line):
                    loop_info['has_logarithmic'] = True
            
            # Detectar fin de loop
            elif re.search(r'\b(end|fin)\b', line):
                if loop_stack:
                    loop_stack.pop()
                    current_depth -= 1
        
        loop_info['nested_depth'] = max_depth
        loop_info['simple_loops'] = loop_info['for_loops'] + loop_info['while_loops']
        
        return loop_info

    def _is_logarithmic_pattern(self, line: str) -> bool:
        """Detecta patrones logarítmicos como división por 2, búsqueda binaria"""
        patterns = [
            r'\/\s*2',  # división por 2
            r'\*\s*2',  # multiplicación por 2
            r'izquierda.*derecha',  # búsqueda binaria
            r'left.*right',
            r'medio',
            r'middle',
            r'>>',  # desplazamiento binario
            r'<<'
        ]
        return any(re.search(pattern, line) for pattern in patterns)

    def _analyze_recursion(self) -> Dict:
        """Analiza recursión y detecta patrones divide y conquista"""
        recursion_info = {
            'has_recursion': False,
            'pattern': None,
            'recurrence': None,
            'recursive_calls': 0
        }
        
        lines = self.code.split('\n')
        function_name = None
        
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
        recursive_calls = 0
        has_base_case = False
        divides_problem = False
        
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

    def _analyze_conditionals(self) -> Dict:
        """Analiza condicionales y su impacto en complejidad"""
        conditional_info = {
            'count': 0,
            'nested': False,
            'affects_omega': False,
            'early_return': False
        }
        
        lines = self.code.split('\n')
        if_depth = 0
        
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

    def _calculate_big_o(self, loop_info: Dict, recursion_info: Dict) -> str:
        """Calcula la complejidad O (peor caso)"""
        
        # Recursión tiene prioridad
        if recursion_info['has_recursion']:
            pattern = recursion_info['pattern']
            if pattern == 'divide_y_conquista_simple':
                return "O(log n)"
            elif pattern == 'divide_y_conquista_doble':
                return "O(n log n)"
            elif pattern == 'recursion_lineal':
                return "O(n)"
            elif pattern == 'recursion_multiple':
                calls = recursion_info['recursive_calls']
                return f"O({calls}^n)"
        
        # Loops logarítmicos
        if loop_info['has_logarithmic']:
            return "O(log n)"
        
        # Loops anidados
        depth = loop_info['nested_depth']
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

    def _calculate_omega(self, loop_info: Dict, recursion_info: Dict, 
                        conditional_info: Dict) -> str:
        """Calcula la complejidad Ω (mejor caso)"""
        
        # Si hay early return, el mejor caso puede ser constante
        if conditional_info['early_return']:
            return "Ω(1)"
        
        # Si hay recursión sin early return
        if recursion_info['has_recursion']:
            # El mejor caso suele ser el caso base
            return "Ω(1)"
        
        # Para loops, el mejor caso es similar al peor caso
        # a menos que haya condicionales con break
        if 'break' in self.code or 'salir' in self.code:
            return "Ω(1)"
        
        # Sin optimizaciones, Ω es igual a O
        big_o = self._calculate_big_o(loop_info, recursion_info)
        return big_o.replace('O', 'Ω')

    def _calculate_theta(self, big_o: str, omega: str, 
                        conditional_info: Dict) -> str:
        """Calcula la complejidad Θ (caso promedio)"""
        
        # Si O y Ω son iguales, entonces Θ es el mismo
        if big_o.replace('O', '') == omega.replace('Ω', ''):
            return big_o.replace('O', 'Θ')
        
        # Si hay diferencia significativa entre mejor y peor caso
        # el caso promedio suele ser el peor caso
        if conditional_info['affects_omega']:
            # Caso promedio es más cercano al peor caso
            return big_o.replace('O', 'Θ')
        
        return big_o.replace('O', 'Θ')

    def _generate_explanation(self, loop_info: Dict, recursion_info: Dict,
                            conditional_info: Dict) -> str:
        """Genera explicación detallada del análisis"""
        explanation = []
        
        if recursion_info['has_recursion']:
            explanation.append(
                f"Algoritmo recursivo detectado ({recursion_info['pattern']})"
            )
            if recursion_info['recurrence']:
                explanation.append(f"Recurrencia: {recursion_info['recurrence']}")
        
        if loop_info['nested_depth'] == 1:
            explanation.append("Loop simple detectado → O(n)")
        elif loop_info['nested_depth'] == 2:
            explanation.append("Loops anidados (2 niveles) detectados → O(n²)")
        elif loop_info['nested_depth'] > 2:
            explanation.append(
                f"Loops anidados ({loop_info['nested_depth']} niveles) → O(n^{loop_info['nested_depth']})"
            )
        
        if loop_info['has_logarithmic']:
            explanation.append("Patrón logarítmico detectado (búsqueda binaria o similar)")
        
        if conditional_info['affects_omega']:
            explanation.append(
                "Condicionales con early return afectan Ω (mejor caso puede ser O(1))"
            )
        else:
            explanation.append(
                "Condicionales no cambian el orden de complejidad"
            )
        
        return " | ".join(explanation)

    def get_complexity(self) -> str:
        """Retorna resumen de complejidades"""
        if not self.complexity_result:
            return "No analysis performed yet"
        
        result = self.complexity_result
        return (
            f"Big-O (peor caso): {result.big_o}\n"
            f"Omega (mejor caso): {result.omega}\n"
            f"Theta (caso promedio): {result.theta}\n"
            f"Explicación: {result.explanation}"
        )

    def get_detailed_analysis(self) -> Dict:
        """Retorna análisis completo en formato diccionario"""
        if not self.complexity_result:
            return {"error": "No analysis performed yet"}
        
        result = self.complexity_result
        return {
            "big_o": result.big_o,
            "omega": result.omega,
            "theta": result.theta,
            "explanation": result.explanation,
            "recurrence": result.recurrence,
            "pattern_type": result.pattern_type
        }