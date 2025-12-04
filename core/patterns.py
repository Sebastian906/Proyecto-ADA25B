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

from typing import Dict, List
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

    def analyze(self, code: str) -> List[Dict]:
        """
        Analiza el código en busca de patrones algorítmicos basados en su estructura.

        Args:
            code (str): Código fuente o pseudocódigo a analizar.

        Returns:
            List[Dict]: Lista de diccionarios con patrones encontrados y detalles.
        """
        self.code = code.lower()
        self.patterns = []

        # Imprimir el código analizado
        print("\nCódigo analizado:")
        print("-" * 50)
        print(code.strip())
        print("-" * 50)

        # Detectar patrones estructurales
        divide_and_conquer_confidence = self._detect_divide_and_conquer()
        if divide_and_conquer_confidence > 0:
            self.patterns.append({
                'name': 'divide_y_conquista',
                'description': 'Divide y Conquista',
                'confidence': divide_and_conquer_confidence
            })

        dynamic_programming_confidence = self._detect_dynamic_programming()
        if dynamic_programming_confidence > 0:
            self.patterns.append({
                'name': 'programacion_dinamica',
                'description': 'Programación Dinámica',
                'confidence': dynamic_programming_confidence
            })

        branch_and_bound_confidence = self._detect_branch_and_bound()
        if branch_and_bound_confidence > 0:
            self.patterns.append({
                'name': 'branch_and_bound',
                'description': 'Branch and Bound',
                'confidence': branch_and_bound_confidence
            })

        graph_algorithms_confidence = self._detect_graph_algorithms()
        if graph_algorithms_confidence > 0:
            self.patterns.append({
                'name': 'algoritmos_grafos',
                'description': 'Algoritmos de Grafos',
                'confidence': graph_algorithms_confidence
            })

        sorting_algorithm_confidence = self._detect_sorting_algorithm()
        if sorting_algorithm_confidence > 0:
            self.patterns.append({
                'name': 'ordenamiento',
                'description': 'Algoritmos de Ordenamiento',
                'confidence': sorting_algorithm_confidence
            })

        return self.patterns

    def _detect_divide_and_conquer(self) -> float:
        """
        Detecta patrones de Divide y Conquista y calcula la confianza basada en las partes clave y su importancia.

        Returns:
            float: Confianza en porcentaje (0.0 a 100.0).
        """
        confidence = 0.0

        # Pesos de las partes clave
        weights = {
            'recursion': 50,  # Recursión es el componente más importante
            'division': 30,   # División del problema es importante
            'combination': 20 # Combinación de resultados es menos importante
        }

        # Detectar recursión
        has_recursion = bool(re.search(r'\breturn\b.*\(', self.code))
        if has_recursion:
            confidence += weights['recursion']

        # Detectar división del problema
        has_division = bool(re.search(r'/ 2|// 2', self.code))
        if has_division:
            confidence += weights['division']

        # Detectar combinación de resultados
        has_combination = bool(re.search(r'merge|combine', self.code))
        if has_combination:
            confidence += weights['combination']

        print(f"Divide y Conquista - Recursión: {has_recursion}, División: {has_division}, Combinación: {has_combination}, Confianza: {confidence:.1f}%")
        return confidence

    def _detect_dynamic_programming(self) -> float:
        """
        Detecta patrones de Programación Dinámica y calcula la confianza basada en las partes clave y su importancia.

        Returns:
            float: Confianza en porcentaje (0.0 a 100.0).
        """
        confidence = 0.0

        # Pesos de las partes clave
        weights = {
            'table': 40,          # Uso de tablas o matrices es muy importante
            'subproblem_access': 40,  # Acceso a subproblemas es muy importante
            'nested_loops': 20    # Bucles anidados son menos importantes
        }

        # Detectar uso de tablas o matrices
        has_table = bool(re.search(r'\[\w+\]', self.code))
        if has_table:
            confidence += weights['table']

        # Detectar acceso a subproblemas
        has_subproblem_access = bool(re.search(r'\[\w+\s*-\s*1\]', self.code))
        if has_subproblem_access:
            confidence += weights['subproblem_access']

        # Detectar bucles anidados
        has_nested_loops = self._detect_nested_loops()
        if has_nested_loops:
            confidence += weights['nested_loops']

        print(f"Programación Dinámica - Tabla: {has_table}, Subproblemas: {has_subproblem_access}, Bucles anidados: {has_nested_loops}, Confianza: {confidence:.1f}%")
        return confidence

    def _detect_greedy_algorithm(self) -> bool:
        # Detectar decisiones óptimas locales
        has_local_optimum = bool(re.search(r'max|min', self.code))
        # Detectar iteración simple
        has_single_iteration = not self._detect_nested_loops()
        return has_local_optimum and has_single_iteration

    def _detect_backtracking(self) -> float:
        """
        Detecta patrones de Backtracking y calcula la confianza basada en las partes clave y su importancia.

        Returns:
            float: Confianza en porcentaje (0.0 a 100.0).
        """
        confidence = 0.0

        # Pesos de las partes clave
        weights = {
            'recursion': 60,  # Recursión es el componente más importante
            'undo': 40        # Deshacer decisiones es importante
        }

        # Detectar recursión
        has_recursion = bool(re.search(r'\breturn\b.*\(', self.code))
        if has_recursion:
            confidence += weights['recursion']

        # Detectar deshacer decisiones
        has_undo = bool(re.search(r'undo|backtrack', self.code))
        if has_undo:
            confidence += weights['undo']

        print(f"Backtracking - Recursión: {has_recursion}, Deshacer decisiones: {has_undo}, Confianza: {confidence:.1f}%")
        return confidence

    def _detect_branch_and_bound(self) -> float:
        """
        Detecta patrones de Branch and Bound y calcula la confianza basada en las partes clave.

        Returns:
            float: Confianza en porcentaje (0.0 a 100.0).
        """
        confidence = 0.0

        # Pesos de las partes clave
        weights = {
            'pruning': 60,  # Poda de soluciones es el componente más importante
            'bounds': 40    # Cálculo de límites es importante
        }

        # Detectar poda de soluciones
        has_pruning = bool(re.search(r'prune', self.code))
        if has_pruning:
            confidence += weights['pruning']

        # Detectar cálculo de límites
        has_bounds = bool(re.search(r'bound', self.code))
        if has_bounds:
            confidence += weights['bounds']

        print(f"Branch and Bound - Poda: {has_pruning}, Límites: {has_bounds}, Confianza: {confidence:.1f}%")
        return confidence

    def _detect_graph_algorithms(self) -> float:
        """
        Detecta patrones de Algoritmos de Grafos y calcula la confianza basada en las partes clave.

        Returns:
            float: Confianza en porcentaje (0.0 a 100.0).
        """
        confidence = 0.0

        # Pesos de las partes clave
        weights = {
            'graph_structure': 50,  # Estructura de grafos es el componente más importante
            'traversal': 50         # Recorrido de grafos es igualmente importante
        }

        # Detectar estructuras de grafos
        has_graph_structure = bool(re.search(r'graph|adjacency', self.code))
        if has_graph_structure:
            confidence += weights['graph_structure']

        # Detectar recorrido de grafos
        has_traversal = bool(re.search(r'bfs|dfs', self.code))
        if has_traversal:
            confidence += weights['traversal']

        print(f"Algoritmos de Grafos - Estructura: {has_graph_structure}, Recorrido: {has_traversal}, Confianza: {confidence:.1f}%")
        return confidence

    def _detect_sorting_algorithm(self) -> float:
        """
        Detecta patrones de Algoritmos de Ordenamiento y calcula la confianza basada en las partes clave.

        Returns:
            float: Confianza en porcentaje (0.0 a 100.0).
        """
        confidence = 0.0

        # Pesos de las partes clave
        weights = {
            'comparisons': 60,  # Comparaciones son el componente más importante
            'swaps': 40         # Intercambios son importantes
        }

        # Detectar comparaciones
        has_comparisons = bool(re.search(r'if.*<|if.*>', self.code))
        if has_comparisons:
            confidence += weights['comparisons']

        # Detectar intercambios
        has_swaps = bool(re.search(r'swap', self.code))
        if has_swaps:
            confidence += weights['swaps']

        print(f"Algoritmos de Ordenamiento - Comparaciones: {has_comparisons}, Intercambios: {has_swaps}, Confianza: {confidence:.1f}%")
        return confidence

    def _detect_nested_loops(self) -> bool:
        # Detectar bucles anidados
        lines = self.code.split('\n')
        loop_stack = []
        for line in lines:
            if re.search(r'for|while', line):
                loop_stack.append(line)
            elif re.search(r'end|}', line):
                if loop_stack:
                    loop_stack.pop()
        return len(loop_stack) > 1

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
            return "No se detectaron patrones algorítmicos.\n"

        summary = ["\nPatrones Algorítmicos Detectados:"]
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
                f"- {pattern['description']}\n"
                f"  Confianza: {numeric_confidence:.1f}%\n"
            )

        # Agregar promedio de confianza
        if pattern_count > 0:
            avg_confidence = total_confidence / pattern_count
            summary.append(f"Confianza promedio de detección: {avg_confidence:.1f}%\n")

            # Recomendaciones basadas en confianza
            if avg_confidence < 50:
                summary.append("Recomendaciones para mejorar la detección:")
                summary.append("- Agregar más documentación en el código")
                summary.append("- Usar nombres de variables más descriptivos")
                summary.append("- Estructurar mejor los bloques de código")
            elif avg_confidence < 70:
                summary.append("Sugerencias para optimizar:")
                summary.append("- Verificar la estructura del algoritmo")
                summary.append("- Considerar agregar comentarios explicativos")

        return "\n".join(summary)
