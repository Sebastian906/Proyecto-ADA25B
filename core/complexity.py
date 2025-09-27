"""
Módulo para análisis de complejidad de algoritmos
"""

class ComplexityAnalyzer:
    """
    Clase para analizar la complejidad de algoritmos a partir de su pseudocódigo
    """
    def __init__(self):
        self.code = None
        self.complexity = None

    def analyze(self, code):
        """
        Analiza la complejidad de un algoritmo dado su pseudocódigo
        
        Args:
            code (str): Pseudocódigo del algoritmo a analizar
            
        Returns:
            str: Notación Big-O de la complejidad del algoritmo
        """
        self.code = code
        # Implementación básica - a completar según los requerimientos específicos
        self.complexity = "O(n)"  # Placeholder
        return self.complexity

    def get_complexity(self):
        """
        Retorna la última complejidad analizada
        
        Returns:
            str: Notación Big-O de la complejidad del algoritmo
        """
        return self.complexity if self.complexity else "No analysis performed yet"
