"""
Módulo para reconocimiento de patrones algorítmicos avanzados
"""

from typing import Dict, List, Tuple
import re

class PatternRecognizer:
    """
    Clase para reconocer patrones comunes en algoritmos
    Incluye detección específica de divide y conquista, patrones iterativos, etc.
    """
    def __init__(self):
        self.code = None
        self.patterns = []
        self.common_patterns = {
            'loop_simple': {
                'keywords': ['for', 'para', 'while', 'mientras'],
                'description': 'Loop simple no anidado',
                'complexity': 'O(n)'
            },
            'loop_anidado': {
                'keywords': ['for.*for', 'para.*para'],
                'description': 'Loops anidados',
                'complexity': 'O(n²) o mayor'
            },
            'recursion_simple': {
                'keywords': ['return.*function', 'return.*funcion'],
                'description': 'Recursión simple',
                'complexity': 'Varía según el caso'
            },
            'divide_y_conquista': {
                'keywords': ['divide', 'merge', 'combine', '/ 2', '// 2'],
                'description': 'Divide y Conquista',
                'complexity': 'O(n log n) típicamente'
            },
            'busqueda_binaria': {
                'keywords': ['izquierda', 'derecha', 'medio', 'left', 'right', 'middle'],
                'description': 'Búsqueda Binaria',
                'complexity': 'O(log n)'
            },
            'programacion_dinamica': {
                'keywords': ['memo', 'table', 'dp', 'cache', 'memorization'],
                'description': 'Programación Dinámica',
                'complexity': 'Varía, generalmente reduce complejidad exponencial'
            },
            'algoritmo_voraz': {
                'keywords': ['sort', 'ordenar', 'select', 'seleccionar', 'choose', 'elegir', 'greedy'],
                'description': 'Algoritmo Voraz (Greedy)',
                'complexity': 'Depende del problema'
            },
            'backtracking': {
                'keywords': ['backtrack', 'try', 'intentar', 'deshacer', 'undo'],
                'description': 'Backtracking',
                'complexity': 'Exponencial en peor caso'
            }
        }

    def analyze(self, code: str) -> List[Dict]:
        """
        Analiza el código en busca de patrones algorítmicos comunes
        
        Args:
            code (str): Código fuente o pseudocódigo a analizar
            
        Returns:
            list: Lista de diccionarios con patrones encontrados y detalles
        """
        self.code = code.lower()
        self.patterns = []

        # Detectar cada patrón
        for pattern_name, pattern_info in self.common_patterns.items():
            if self._detect_pattern(pattern_name, pattern_info):
                self.patterns.append({
                    'name': pattern_name,
                    'description': pattern_info['description'],
                    'complexity': pattern_info['complexity'],
                    'confidence': self._calculate_confidence(pattern_name)
                })
        
        # Análisis específico de estructura
        self._analyze_loop_structure()
        self._analyze_recursion_pattern()
        
        return self.patterns
    
    def _detect_pattern(self, pattern_name: str, pattern_info: Dict) -> bool:
        """Detecta si un patrón específico está presente"""
        keywords = pattern_info['keywords']
        
        # Para patrones regex
        if pattern_name == 'loop_anidado':
            return self._detect_nested_loops()
        elif pattern_name == 'busqueda_binaria':
            return self._detect_binary_search()
        elif pattern_name == 'divide_y_conquista':
            return self._detect_divide_conquer()
        
        # Detección simple por palabras clave
        return any(
            re.search(rf'\b{keyword}\b', self.code) 
            for keyword in keywords
        )

    def _detect_nested_loops(self) -> bool:
        """Detecta loops anidados específicamente"""
        lines = self.code.split('\n')
        loop_stack = []
        max_depth = 0
        
        for line in lines:
            line_clean = line.strip()
            
            # Detectar inicio de loop (completo con HACER/DO)
            if re.search(r'\b(for|para|while|mientras)\b.*\b(hacer|do)\b', line_clean, re.IGNORECASE):
                loop_stack.append(line_clean)
                max_depth = max(max_depth, len(loop_stack))
            
            # Detectar fin de loop
            elif re.search(r'\b(fin|end)\b\s*(para|for|mientras|while)?\b', line_clean, re.IGNORECASE):
                if loop_stack:
                    loop_stack.pop()
        
        return max_depth >= 2

    def _detect_binary_search(self) -> bool:
        """Detecta patrón específico de búsqueda binaria"""
        has_left_right = bool(re.search(r'(izquierda|left).*(derecha|right)', self.code))
        has_middle = bool(re.search(r'\b(medio|middle)\b', self.code))
        has_division = bool(re.search(r'\/\s*2|\/\/\s*2', self.code))
        
        return (has_left_right and has_middle) or (has_division and has_middle)

    def _detect_divide_conquer(self) -> bool:
        """Detecta patrón divide y conquista"""
        # Buscar recursión + división del problema
        has_recursion = bool(re.search(r'\breturn\b.*\(', self.code))
        has_division = bool(re.search(r'\/\s*2|divide|split|merge', self.code))
        
        # Buscar palabras clave específicas
        dc_keywords = ['merge', 'combine', 'divide', 'conquer', 'conquistar']
        has_keywords = any(keyword in self.code for keyword in dc_keywords)
        
        return (has_recursion and has_division) or has_keywords

    def _analyze_loop_structure(self):
        """Analiza estructura de loops en detalle"""
        lines = self.code.split('\n')
        loop_stack = []
        loop_types = []
        
        for line in lines:
            line_clean = line.strip()
            
            # Detectar inicio de loop (completo con HACER/DO)
            if re.search(r'\b(for|para)\b.*\b(hacer|do)\b', line_clean, re.IGNORECASE):
                loop_stack.append('for')
                loop_types.append('for')
            elif re.search(r'\b(while|mientras)\b.*\b(hacer|do)\b', line_clean, re.IGNORECASE):
                loop_stack.append('while')
                loop_types.append('while')
            
            # Detectar fin de loop
            elif re.search(r'\b(fin|end)\b\s*(para|for|mientras|while)?\b', line_clean, re.IGNORECASE):
                if loop_stack:
                    loop_stack.pop()
        
        loop_count = len(loop_types)
        max_depth = 0
        current_depth = 0
        
        # Recalcular profundidad máxima
        for line in lines:
            line_clean = line.strip()
            if re.search(r'\b(for|para|while|mientras)\b.*\b(hacer|do)\b', line_clean, re.IGNORECASE):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif re.search(r'\b(fin|end)\b\s*(para|for|mientras|while)?\b', line_clean, re.IGNORECASE):
                current_depth = max(0, current_depth - 1)
        
        if loop_count > 0:
            # Actualizar o agregar información de loops
            loop_info = {
                'name': 'loop_analysis',
                'description': f'{loop_count} loop(s) detectado(s): {", ".join(loop_types)}',
                'complexity': self._estimate_loop_complexity(max_depth),
                'confidence': 'high'
            }
            
            # Evitar duplicados
            if not any(p['name'] == 'loop_analysis' for p in self.patterns):
                self.patterns.append(loop_info)

    def _analyze_recursion_pattern(self):
        """Analiza patrón de recursión en detalle"""
        lines = self.code.split('\n')
        function_name = None
        recursive_calls = 0
        
        # Buscar función
        for line in lines:
            if re.search(r'\b(function|funcion|def|procedimiento)\b', line):
                match = re.search(r'\b(function|funcion|def|procedimiento)\s+(\w+)', line)
                if match:
                    function_name = match.group(2)
                    break
        
        # Contar llamadas recursivas
        if function_name:
            for line in lines:
                if re.search(rf'\b{function_name}\s*\(', line):
                    recursive_calls += 1
        
        if recursive_calls > 0:
            recursion_info = {
                'name': 'recursion_analysis',
                'description': f'Recursión con {recursive_calls} llamada(s) recursiva(s)',
                'complexity': self._estimate_recursion_complexity(recursive_calls),
                'confidence': 'high'
            }
            
            if not any(p['name'] == 'recursion_analysis' for p in self.patterns):
                self.patterns.append(recursion_info)

    def _estimate_loop_complexity(self, depth: int) -> str:
        """Estima complejidad basada en número de loops"""
        if depth == 0:
            return "O(1)"
        elif depth == 1:
            return "O(n)"
        elif depth == 2:
            return "O(n²)"
        else:
            return f"O(n^{depth})"

    def _estimate_recursion_complexity(self, recursive_calls: int) -> str:
        """Estima complejidad de recursión"""
        if recursive_calls == 1:
            if self._detect_divide_conquer():
                return "O(log n) o O(n log n)"
            return "O(n)"
        else:
            return f"O({recursive_calls}^n)"

    def _calculate_confidence(self, pattern_name: str) -> str:
        """Calcula nivel de confianza de detección"""
        # Patrones más confiables
        high_confidence = ['busqueda_binaria', 'loop_simple']
        medium_confidence = ['divide_y_conquista', 'programacion_dinamica']
        
        if pattern_name in high_confidence:
            return 'high'
        elif pattern_name in medium_confidence:
            return 'medium'
        else:
            return 'low'

    def get_patterns(self) -> List[Dict]:
        """
        Retorna los patrones encontrados en el último análisis
        
        Returns:
            list: Lista de diccionarios con patrones y detalles
        """
        return self.patterns if self.patterns else []

    def add_pattern(self, pattern_name: str, keywords: List[str], 
                    description: str = "", complexity: str = ""):
        """
        Agrega un nuevo patrón para reconocimiento
        
        Args:
            pattern_name (str): Nombre del nuevo patrón
            keywords (list): Lista de palabras clave asociadas al patrón
            description (str): Descripción del patrón
            complexity (str): Complejidad típica del patrón
        """
        if pattern_name and keywords:
            self.common_patterns[pattern_name] = {
                'keywords': keywords,
                'description': description or f'Patrón personalizado: {pattern_name}',
                'complexity': complexity or 'No especificada'
            }

    def get_summary(self) -> str:
        """Retorna un resumen legible de los patrones encontrados"""
        if not self.patterns:
            return "No se detectaron patrones algorítmicos"
        
        summary = ["Patrones algorítmicos detectados:\n"]
        for pattern in self.patterns:
            summary.append(
                f"- {pattern['description']} "
                f"(Complejidad: {pattern['complexity']}, "
                f"Confianza: {pattern['confidence']})"
            )
        
        return "\n".join(summary)

    def detect_divide_and_conquer_detail(self) -> Dict:
        """
        Análisis detallado específico para Divide y Conquista
        
        Returns:
            dict: Información detallada sobre el patrón D&C si existe
        """
        if not self._detect_divide_conquer():
            return {"detected": False}
        
        lines = self.code.split('\n')
        
        analysis = {
            "detected": True,
            "has_base_case": False,
            "has_recursive_calls": 0,
            "divides_by_two": False,
            "has_merge_step": False
        }
        
        for line in lines:
            if re.search(r'\breturn\b.*[0-9]', line):
                analysis["has_base_case"] = True
            if re.search(r'\breturn\b.*\(', line):
                analysis["has_recursive_calls"] += 1
            if re.search(r'\/\s*2|\/\/\s*2', line):
                analysis["divides_by_two"] = True
            if re.search(r'\bmerge\b|\bcombine\b|\bcombinar\b', line):
                analysis["has_merge_step"] = True
        
        # Determinar tipo específico
        if analysis["has_recursive_calls"] == 2 and analysis["has_merge_step"]:
            analysis["type"] = "merge_sort_like"
            analysis["expected_complexity"] = "O(n log n)"
        elif analysis["has_recursive_calls"] == 1 and analysis["divides_by_two"]:
            analysis["type"] = "binary_search_like"
            analysis["expected_complexity"] = "O(log n)"
        else:
            analysis["type"] = "generic_divide_conquer"
            analysis["expected_complexity"] = "Depende del caso específico"
        
        return analysis