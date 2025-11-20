"""
Módulo `patterns.py`

Este módulo se encarga de reconocer patrones algorítmicos comunes y avanzados en pseudocódigo o código fuente.
Incluye detección específica de patrones como:
- Divide y conquista
- Programación dinámica
- Algoritmos voraces (Greedy)
- Backtracking
- Branch and Bound
- Algoritmos de grafos (BFS, DFS, Dijkstra, etc.)
- Algoritmos de ordenamiento (QuickSort, MergeSort, etc.)
"""

from typing import Dict, List, Tuple
import re


class PatternRecognizer:
    """
    Clase para reconocer patrones comunes en algoritmos.

    Propósito:
        Detectar patrones algorítmicos en pseudocódigo o código fuente, como divide y conquista,
        programación dinámica, algoritmos voraces, backtracking, branch and bound, y algoritmos de grafos.

    Métodos principales:
        - analyze(code: str) -> List[Dict]: Analiza el código en busca de patrones algorítmicos comunes.
        - add_pattern(pattern_name: str, keywords: List[str], description: str, complexity: str): Agrega un nuevo patrón para reconocimiento.
        - get_patterns() -> List[Dict]: Retorna los patrones encontrados en el último análisis.
        - get_summary() -> str: Retorna un resumen legible de los patrones encontrados.
    """

    def __init__(self):
        self.code = None
        self.patterns = []
        self.common_patterns = {
            'divide_y_conquista': {
                'keywords': ['divide', 'merge', 'combine', '/ 2', '// 2'],
                'description': 'Divide y Conquista',
                'complexity': 'O(n log n) típicamente'
            },
            'programacion_dinamica': {
                'keywords': ['memo', 'table', 'dp', 'cache', 'subproblem'],
                'description': 'Programación Dinámica',
                'complexity': 'Varía, generalmente reduce complejidad exponencial'
            },
            'algoritmo_voraz': {
                'keywords': ['greedy', 'select', 'optimal', 'local'],
                'description': 'Algoritmo Voraz (Greedy)',
                'complexity': 'Depende del problema'
            },
            'backtracking': {
                'keywords': ['backtrack', 'try', 'undo', 'solution'],
                'description': 'Backtracking',
                'complexity': 'Exponencial en peor caso'
            },
            'branch_and_bound': {
                'keywords': ['bound', 'branch', 'prune', 'tree'],
                'description': 'Branch and Bound',
                'complexity': 'Exponencial en peor caso'
            },
            'algoritmos_grafos': {
                'keywords': ['bfs', 'dfs', 'dijkstra', 'prim', 'kruskal'],
                'description': 'Algoritmos de Grafos',
                'complexity': 'Varía según el algoritmo'
            },
            'ordenamiento': {
                'keywords': ['quicksort', 'mergesort', 'bubble', 'insertion', 'selection'],
                'description': 'Algoritmos de Ordenamiento',
                'complexity': 'Varía según el algoritmo'
            }
        }

    def analyze(self, code: str) -> List[Dict]:
        """
        Analiza el código en busca de patrones algorítmicos comunes.

        Args:
            code (str): Código fuente o pseudocódigo a analizar.

        Returns:
            List[Dict]: Lista de diccionarios con patrones encontrados y detalles, incluyendo:
                - name (str): Nombre del patrón.
                - description (str): Descripción del patrón.
                - complexity (str): Complejidad típica del patrón.
                - confidence (float): Nivel de confianza en la detección (en porcentaje).
        """
        self.code = code.lower()
        self.patterns = []

        # Detectar cada patrón
        for pattern_name, pattern_info in self.common_patterns.items():
            if self._detect_pattern(pattern_name, pattern_info):
                confidence = self._calculate_confidence(pattern_name)
                self.patterns.append({
                    'name': pattern_name,
                    'description': pattern_info['description'],
                    'complexity': pattern_info['complexity'],
                    'confidence': confidence
                })

        return self.patterns

    def _detect_pattern(self, pattern_name: str, pattern_info: Dict) -> bool:
        """
        Detecta si un patrón específico está presente en el código.

        Args:
            pattern_name (str): Nombre del patrón a detectar.
            pattern_info (Dict): Información del patrón, incluyendo palabras clave y descripción.

        Returns:
            bool: True si el patrón es detectado, False en caso contrario.
        """
        keywords = pattern_info['keywords']
        for keyword in keywords:
            if re.search(rf'\b{keyword}\b', self.code):
                print(f"Patrón detectado: {pattern_name}, palabra clave: {keyword}")
                return True
        return False

    def _calculate_confidence(self, pattern_name: str) -> float:
        """
        Calcula el nivel de confianza en la detección de un patrón.

        La confianza se calcula como el porcentaje de palabras clave detectadas
        en el código en relación con el total de palabras clave asociadas al patrón.
        Además, se asigna un 100% de confianza si el nombre del patrón coincide
        con el nombre de una función en el código.

        Args:
            pattern_name (str): Nombre del patrón detectado.

        Returns:
            float: Nivel de confianza en porcentaje (0.0 a 100.0).
        """
        pattern_info = self.common_patterns[pattern_name]
        keywords = pattern_info['keywords']
        matches = sum(1 for keyword in keywords if keyword in self.code)
        total_keywords = len(keywords)

        # Depuración: Imprimir palabras clave detectadas
        print(f"Patrón: {pattern_name}")
        print(f"Palabras clave: {keywords}")
        print(f"Palabras detectadas: {[keyword for keyword in keywords if keyword in self.code]}")
        print(f"Confianza calculada: {(matches / total_keywords) * 100 if total_keywords > 0 else 0.0}%")

        if total_keywords == 0:
            return 0.0

        # Ajustar confianza con un peso adicional para palabras clave críticas
        critical_keywords = ['merge', 'divide', 'combine', 'dp', 'greedy', 'backtrack']
        critical_matches = sum(1 for keyword in critical_keywords if keyword in self.code)
        confidence_percentage = (matches / total_keywords) * 100

        # Incrementar confianza si se detectan palabras clave críticas
        if critical_matches > 0:
            confidence_percentage += 10 * critical_matches  # 10% por cada palabra clave crítica detectada

        # Regla adicional: Si el nombre del patrón coincide con el nombre de una función, asignar 100%
        function_names = re.findall(r'function\s+(\w+)', self.code)
        if pattern_name in function_names:
            print(f"El nombre del patrón '{pattern_name}' coincide con el nombre de una función. Asignando 100% de confianza.")
            return 100.0

        # Limitar la confianza al 100%
        return min(confidence_percentage, 100.0)

    def get_summary(self) -> str:
        """
        Retorna un resumen legible de los patrones encontrados.

        Returns:
            str: Resumen de los patrones detectados.
        """
        if not self.patterns:
            return "No se detectaron patrones algorítmicos."

        summary = ["Patrones algorítmicos detectados:\n"]
        for pattern in self.patterns:
            summary.append(
                f"- {pattern['description']} (Complejidad: {pattern['complexity']}, "
                f"Confianza: {pattern['confidence']:.1f}%)"
            )

        # Calcular confianza promedio
        avg_confidence = sum(p['confidence'] for p in self.patterns) / len(self.patterns)
        summary.append(f"\nConfianza promedio de detección: {avg_confidence:.1f}%")

        # Recomendaciones
        summary.append("\nRecomendaciones para mejorar la detección:")
        summary.append("- Agregar más documentación en el código")
        summary.append("- Usar nombres de variables más descriptivos")
        summary.append("- Estructurar mejor los bloques de código")

        return "\n".join(summary)
    
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
        """
        Detecta patrones de divide y conquista en el código.

        Returns:
            bool: True si se detecta el patrón de divide y conquista, False en caso contrario.
        """
        # Buscar recursión + división del problema
        has_recursion = bool(re.search(r'\breturn\b.*\(', self.code))
        has_division = bool(re.search(r'\/\s*2|divide|split|merge', self.code))
        
        # Buscar palabras clave específicas
        dc_keywords = ['merge', 'combine', 'divide', 'conquer', 'conquistar', 'mezclar', 'dividir']
        has_keywords = any(keyword in self.code.lower() for keyword in dc_keywords)
        
        # Análisis de estructura
        has_recursive_split = False
        has_merge_step = False
        lines = self.code.split('\n')
        for i, line in enumerate(lines):
            if '/ 2' in line or '// 2' in line:
                # Buscar llamada recursiva después de la división
                for j in range(i+1, min(i+5, len(lines))):
                    if any(kw in lines[j].lower() for kw in ['return', 'llamar', 'recursivo']):
                        has_recursive_split = True
                        break
            if any(kw in line.lower() for kw in ['merge', 'mezclar', 'combinar']):
                has_merge_step = True
        
        base_score = (has_recursion and has_division) or has_keywords
        structure_score = has_recursive_split and has_merge_step
        
        return base_score or structure_score
        
    def _detect_simple_recursion(self) -> bool:
        """Detecta recursión simple con análisis mejorado"""
        # Detectar llamada recursiva básica
        has_recursion = bool(re.search(r'\breturn\b.*\(', self.code))
        
        # Detectar caso base
        has_base_case = bool(re.search(r'if.*(\<=|\>=|==).*[0-9].*return', self.code))
        
        # Detectar patrones comunes de recursión
        recursive_patterns = [
            r'return.*\+.*\(',  # Suma recursiva
            r'return.*\*.*\(',  # Multiplicación recursiva
            r'n\s*-\s*1',      # Decremento típico
            r'factorial',       # Factorial
            r'fibonacci'        # Fibonacci
        ]
        
        pattern_matches = any(bool(re.search(pattern, self.code)) for pattern in recursive_patterns)
        
        # Analizar estructura
        lines = self.code.split('\n')
        depth = 0
        max_depth = 0
        has_proper_structure = False
        
        for line in lines:
            if 'if' in line or 'si' in line.lower():
                depth += 1
            elif 'fin' in line.lower() or 'end' in line.lower():
                depth -= 1
            max_depth = max(depth, max_depth)
            
        has_proper_structure = max_depth >= 1 and depth == 0
        
        return has_recursion and has_base_case and (pattern_matches or has_proper_structure)
        
    def _detect_dynamic_programming(self) -> bool:
        """Detecta programación dinámica con análisis mejorado"""
        # Detectar uso de array/matriz para almacenar resultados
        has_array = bool(re.search(r'(array|arreglo|matriz|table|tabla)\[\w+\]', self.code))
        
        # Detectar inicialización de estructura
        has_initialization = bool(re.search(r'(array|arreglo|matriz|table|tabla).*=.*new', self.code))
        
        # Detectar patrones de acceso a subproblemas
        subproblem_patterns = [
            r'\[\w+\s*-\s*1\]',  # Acceso a elemento anterior
            r'\[\w+\s*-\s*2\]',  # Acceso a dos elementos anteriores
            r'min\(.*\[.*\]',     # Minimización con acceso a array
            r'max\(.*\[.*\]'      # Maximización con acceso a array
        ]
        
        has_subproblem_access = any(bool(re.search(pattern, self.code)) for pattern in subproblem_patterns)
        
        # Detectar bucles característicos de DP
        has_dp_loops = False
        lines = self.code.split('\n')
        in_loop = False
        array_access_in_loop = False
        
        for line in lines:
            if ('for' in line or 'para' in line.lower()) and not in_loop:
                in_loop = True
            elif in_loop and ('[' in line and ']' in line):
                array_access_in_loop = True
            elif ('fin' in line.lower() or 'end' in line.lower()) and in_loop:
                if array_access_in_loop:
                    has_dp_loops = True
                in_loop = False
                array_access_in_loop = False
                
        return (has_array and has_initialization and has_subproblem_access) or \
               (has_array and has_dp_loops and has_subproblem_access)
               
    def _detect_greedy_algorithm(self) -> bool:
        """Detecta algoritmo voraz con análisis mejorado"""
        # Detectar ordenamiento inicial (común en greedy)
        has_sort = bool(re.search(r'(sort|ordenar|order|clasificar)', self.code.lower()))
        
        # Detectar selección de óptimo local
        local_opt_patterns = [
            r'(max|maximum|maximo|mayor)',
            r'(min|minimum|minimo|menor)',
            r'(first|primero|siguiente)',
            r'(optimal|optimo|mejor)'
        ]
        
        has_local_decision = any(bool(re.search(pattern, self.code.lower())) 
                               for pattern in local_opt_patterns)
        
        # Detectar iteración simple (no necesita revisar decisiones previas)
        has_single_pass = self._detect_nested_loops() == False and \
                         bool(re.search(r'(for|while|para|mientras)', self.code.lower()))
                         
        # Detectar patrones de selección voraz
        greedy_patterns = [
            r'(select|seleccionar|choose|elegir)',
            r'(add|agregar|push|append)',
            r'(remove|remover|pop|delete)'
        ]
        
        has_selection = any(bool(re.search(pattern, self.code.lower())) 
                          for pattern in greedy_patterns)
        
        # Análisis de estructura
        has_greedy_structure = False
        lines = self.code.split('\n')
        decision_count = 0
        
        for i, line in enumerate(lines):
            if any(p in line.lower() for p in ['if', 'si']):
                # Verificar si la decisión usa un óptimo local
                for pattern in local_opt_patterns:
                    if bool(re.search(pattern, line.lower())):
                        decision_count += 1
                        break
                        
        has_greedy_structure = decision_count > 0 and decision_count <= 3  # Típicamente pocas decisiones
        
        return (has_sort and has_local_decision and has_single_pass) or \
               (has_selection and has_local_decision and has_greedy_structure)
               
    def _get_context_around_keyword(self, keyword: str, window: int = 100) -> str:
        """Obtiene el contexto alrededor de una palabra clave"""
        match = re.search(rf'\b{keyword}\b', self.code.lower())
        if match:
            start = max(0, match.start() - window)
            end = min(len(self.code), match.end() + window)
            return self.code[start:end]
        return ""
        
    def _analyze_keyword_context(self, context: str, pattern_name: str) -> float:
        """Analiza el contexto alrededor de una palabra clave para mejorar la detección"""
        context = context.lower()
        score = 0.0
        
        # Palabras clave relacionadas por tipo de patrón
        related_terms = {
            'loop_simple': ['inicio', 'fin', 'incremento', 'paso', 'contador'],
            'recursion_simple': ['base', 'caso', 'return', 'llamada', 'recursivo'],
            'backtracking': ['vuelta', 'probar', 'intentar', 'regresar', 'solución'],
            'algoritmo_voraz': ['óptimo', 'mejor', 'selección', 'elegir', 'solución']
        }
        
        # Verificar palabras relacionadas en el contexto
        if pattern_name in related_terms:
            terms = related_terms[pattern_name]
            for term in terms:
                if term in context:
                    score += 0.2  # 20% por cada término relacionado encontrado
                    
        # Verificar estructura típica
        if ('si' in context or 'if' in context) and ('entonces' in context or 'then' in context):
            score += 0.1
        if ('para' in context or 'for' in context) and ('hacer' in context or 'do' in context):
            score += 0.1
            
        return min(1.0, score)

    def _analyze_loop_structure(self):
        """Analiza estructura de loops en detalle con análisis semántico"""
        lines = self.code.split('\n')
        loop_stack = []
        loop_types = []
        loop_variables = set()
        loop_operations = []
        current_depth = 0
        max_depth = 0
        
        for line in lines:
            line_clean = line.strip().lower()
            
            # Detectar inicio de loop con análisis de variables
            for_match = re.search(r'\b(for|para)\s+(\w+)\s+', line_clean)
            while_match = re.search(r'\b(while|mientras)\b.*\b(hacer|do)\b', line_clean)
            
            if for_match:
                loop_var = for_match.group(2)
                loop_variables.add(loop_var)
                loop_stack.append(('for', loop_var))
                loop_types.append('for')
                current_depth += 1
                
                # Analizar rango del loop
                if 'hasta' in line_clean or 'to' in line_clean:
                    range_match = re.search(r'hasta\s+(\w+)', line_clean) or re.search(r'to\s+(\w+)', line_clean)
                    if range_match:
                        range_var = range_match.group(1)
                        if range_var.lower() == 'n':
                            loop_operations.append('n_iteration')
                        elif re.match(r'\d+', range_var):
                            loop_operations.append('constant_iteration')
                
            elif while_match:
                loop_stack.append(('while', None))
                loop_types.append('while')
                current_depth += 1
                
                # Analizar condición del while
                if '/2' in line_clean or '>>1' in line_clean:
                    loop_operations.append('logarithmic')
                elif '<' in line_clean or '>' in line_clean:
                    loop_operations.append('linear')
            
            # Detectar fin de loop
            elif re.search(r'\b(fin|end)\b\s*(para|for|mientras|while)?\b', line_clean, re.IGNORECASE):
                if loop_stack:
                    loop_stack.pop()
                    current_depth = max(0, current_depth - 1)
            
            # Analizar operaciones dentro del loop
            elif loop_stack:
                # Detectar operaciones características
                if re.search(r'\/=\s*2|>>=\s*1', line_clean):
                    loop_operations.append('logarithmic')
                elif re.search(r'\+=|\-=|\*=|\/=', line_clean):
                    loop_operations.append('arithmetic')
                elif 'swap' in line_clean or 'intercambiar' in line_clean:
                    loop_operations.append('swap')
                
                # Analizar acceso a arrays/matrices
                if re.search(r'\[\w+\]', line_clean):
                    loop_operations.append('array_access')
                    if len(re.findall(r'\[\w+\]', line_clean)) > 1:
                        loop_operations.append('matrix_operation')
            
            max_depth = max(max_depth, current_depth)
        
        loop_count = len(loop_types)
        
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

    def _calculate_confidence(self, pattern_name: str) -> float:
        """Calcula nivel de confianza de detección con puntuación numérica mejorada"""
        confidence_score = 0.0
        
        # Análisis del código
        if self.code:
            # Verificar estructura clara y asignar puntuación base
            if pattern_name == 'loop_simple':
                loop_matches = re.findall(r'\b(for|para|while|mientras)\b.*\b(hacer|do)\b', self.code)
                if len(loop_matches) == 1:
                    confidence_score += 50  # Base score para loop simple bien estructurado
                    if not self._detect_nested_loops():
                        confidence_score += 30  # Bonus por confirmar que no hay anidación
                    # Verificar estructura completa
                    if re.search(r'\b(fin|end)\s*(para|for|mientras|while)\b', self.code, re.IGNORECASE):
                        confidence_score += 10  # Bonus por cierre correcto
                    # Verificar variables de control
                    if re.search(r'\b(i|j|k|n)\b\s*(=|<-)\s*\d+', self.code):
                        confidence_score += 10  # Bonus por inicialización correcta
                
            elif pattern_name == 'busqueda_binaria':
                binary_search_score = 0
                if bool(re.search(r'(izquierda|left).*(derecha|right)', self.code)):
                    binary_search_score += 30
                if bool(re.search(r'\b(medio|middle)\b', self.code)):
                    binary_search_score += 30
                if bool(re.search(r'\/\s*2|\/\/\s*2', self.code)):
                    binary_search_score += 20
                confidence_score = binary_search_score
                # Bonus adicional para búsqueda binaria
                if bool(re.search(r'ordenado|sorted|ascendente|ascending', self.code.lower())):
                    confidence_score += 10  # Bonus por mencionar prerrequisito de orden
                if bool(re.search(r'comparar|compare|igual|equal', self.code.lower())):
                    confidence_score += 10  # Bonus por operaciones de comparación
                
            elif pattern_name == 'divide_y_conquista':
                dc_score = 0
                # Análisis base
                if bool(re.search(r'\breturn\b.*\(', self.code)):  # Recursión
                    dc_score += 30
                if bool(re.search(r'\/\s*2|divide|split', self.code)):  # División
                    dc_score += 30
                if bool(re.search(r'merge|combine|conquistar|mezclar', self.code)):  # Combinación
                    dc_score += 20
                
                # Análisis avanzado
                if bool(re.search(r'izquierda|left|derecha|right|mitad|middle', self.code.lower())):
                    dc_score += 10  # Bonus por división explícita
                if bool(re.search(r'if.*return.*[0-9]', self.code)):  # Caso base
                    dc_score += 10  # Bonus por caso base explícito
                confidence_score = dc_score
                
            elif pattern_name == 'programacion_dinamica':
                dp_score = 0
                # Características básicas de DP
                if bool(re.search(r'\b(array|arreglo|matriz|table|tabla)\b', self.code)):
                    dp_score += 25
                if bool(re.search(r'\b(memo|cache|dp)\b', self.code)):
                    dp_score += 35
                if self._detect_nested_loops():  # Común en DP
                    dp_score += 20
                    
                # Análisis avanzado de DP
                if bool(re.search(r'\[\w+\s*-\s*1\]|\[\w+\s*-\s*2\]', self.code)):
                    dp_score += 10  # Acceso a subproblemas
                if bool(re.search(r'min|max|minimum|maximum', self.code.lower())):
                    dp_score += 5  # Optimización típica
                if bool(re.search(r'for.*for.*\[', self.code)):
                    dp_score += 5  # Acceso matricial
                confidence_score = dp_score            # Análisis de coherencia estructural
            lines = self.code.split('\n')
            indentation_pattern = True
            bracket_balance = True
            
            # Verificar indentación consistente
            current_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    if indent % 2 != 0 and indent % 4 != 0:
                        indentation_pattern = False
                        break
            
            # Verificar balance de estructuras
            open_structures = len(re.findall(r'\b(for|while|if|function)\b', self.code))
            close_structures = len(re.findall(r'\b(fin|end)\b', self.code))
            if open_structures != close_structures:
                bracket_balance = False
            
            # Ajustar confianza basado en estructura
            if indentation_pattern and bracket_balance:
                confidence_score = min(100, confidence_score + 10)
            elif not indentation_pattern or not bracket_balance:
                confidence_score = max(0, confidence_score - 20)
        
        # Normalizar score a porcentaje
        confidence_score = max(0, min(100, confidence_score))
        
        # Convertir a niveles tradicionales para compatibilidad
        if confidence_score >= 75:
            return 'high'
        elif confidence_score >= 45:
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
        total_confidence = 0
        pattern_count = 0
        
        for pattern in self.patterns:
            confidence_level = pattern['confidence']
            # Convertir nivel de confianza a número si es string
            if isinstance(confidence_level, str):
                if confidence_level == 'high':
                    numeric_confidence = 85
                elif confidence_level == 'medium':
                    numeric_confidence = 60
                elif confidence_level == 'low':
                    numeric_confidence = 35
            else:
                numeric_confidence = float(confidence_level)
                
            total_confidence += numeric_confidence
            pattern_count += 1
            
            summary.append(
                f"- {pattern['description']} "
                f"(Complejidad: {pattern['complexity']}, "
                f"Confianza: {numeric_confidence:.1f}%)"
            )
        
        # Agregar promedio de confianza
        if pattern_count > 0:
            avg_confidence = total_confidence / pattern_count
            summary.append(f"\nConfianza promedio de detección: {avg_confidence:.1f}%")
            
            # Recomendaciones basadas en confianza
            if avg_confidence < 50:
                summary.append("\nRecomendaciones para mejorar la detección:")
                summary.append("- Agregar más documentación en el código")
                summary.append("- Usar nombres de variables más descriptivos")
                summary.append("- Estructurar mejor los bloques de código")
            elif avg_confidence < 70:
                summary.append("\nSugerencias para optimizar:")
                summary.append("- Verificar la estructura del algoritmo")
                summary.append("- Considerar agregar comentarios explicativos")
        
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