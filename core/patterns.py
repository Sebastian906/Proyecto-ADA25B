"""
Módulo para reconocimiento de patrones en algoritmos
"""

class PatternRecognizer:
    """
    Clase para reconocer patrones comunes en algoritmos
    """
    def __init__(self):
        self.code = None
        self.patterns = []
        self.common_patterns = {
            'loop': ['for', 'while', 'do'],
            'recursion': ['return', 'self'],
            'divide_conquer': ['divide', 'merge', 'combine'],
            'dynamic': ['memo', 'table', 'dp'],
            'greedy': ['sort', 'select', 'choose']
        }

    def analyze(self, code):
        """
        Analiza el código en busca de patrones algorítmicos comunes
        
        Args:
            code (str): Código fuente o pseudocódigo a analizar
            
        Returns:
            list: Lista de patrones encontrados en el código
        """
        self.code = code
        self.patterns = []
        # Implementación básica - a completar según requerimientos específicos
        # Por ahora solo detecta la presencia de palabras clave
        for pattern, keywords in self.common_patterns.items():
            if any(keyword in code.lower() for keyword in keywords):
                self.patterns.append(pattern)
        return self.patterns

    def get_patterns(self):
        """
        Retorna los patrones encontrados en el último análisis
        
        Returns:
            list: Lista de patrones encontrados
        """
        return self.patterns if self.patterns else []

    def add_pattern(self, pattern_name, keywords):
        """
        Agrega un nuevo patrón para reconocimiento
        
        Args:
            pattern_name (str): Nombre del nuevo patrón
            keywords (list): Lista de palabras clave asociadas al patrón
        """
        if pattern_name and keywords:
            self.common_patterns[pattern_name] = keywords
