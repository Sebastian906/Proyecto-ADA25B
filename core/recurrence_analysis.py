"""
Módulo para el análisis de recurrencias.

Este módulo analiza el pseudocódigo línea por línea, genera las funciones temporales
y espaciales, y resuelve la ecuación de recurrencia mostrando el proceso paso a paso.
"""

from typing import List, Dict, Tuple
import sympy as sp

def analyze_recurrence(code_lines: List[Dict]) -> Dict:
    """
    Analiza las recurrencias en el pseudocódigo.

    Args:
        code_lines: Lista de líneas analizadas con sus costos temporales.

    Returns:
        Un diccionario con:
        - La ecuación de recurrencia.
        - El desglose de las series.
        - La solución final.
    """
    # Variables simbólicas
    n = sp.Symbol('n')
    T = sp.Function('T')

    # Inicializar ecuación de recurrencia
    recurrence = 0
    series_steps = []

    for line in code_lines:
        time_cost = line.get('time_cost', 0)
        exec_count = line.get('exec_count', 1)

        # Generar función temporal de la línea
        line_cost = time_cost * exec_count
        series_steps.append(f"{line['line']}: {line_cost}")

        # Sumar al total de la recurrencia
        recurrence += line_cost

    # Resolver la ecuación de recurrencia
    recurrence_eq = sp.Eq(T(n), recurrence)
    solution = sp.rsolve(recurrence_eq, T(n))

    return {
        "recurrence_eq": str(recurrence_eq),
        "series_steps": series_steps,
        "solution": str(solution)
    }