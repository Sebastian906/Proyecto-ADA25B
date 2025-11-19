"""
Módulo `integration.py`

Este módulo se encarga de la integración con la API de Gemini para enviar prompts y recibir respuestas. 
Incluye un fallback local que utiliza los analizadores del proyecto para generar respuestas heurísticas 
cuando la API no está disponible.

Funciones principales:
    - ask_gemini: Envía un prompt a Gemini y retorna la respuesta.
    - fallback_ask_gemini: Genera una respuesta heurística utilizando los analizadores locales.
"""

import warnings
import os
import logging

# Suprimir warnings y logs de Google y gRPC
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurar logging para suprimir mensajes de dependencias
logging.getLogger('google').setLevel(logging.CRITICAL)
logging.getLogger('grpc').setLevel(logging.CRITICAL)
logging.getLogger('gax').setLevel(logging.CRITICAL)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv no disponible en el entorno de pruebas; continuar sin él
    pass

try:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def ask_gemini(prompt: str) -> str:
        """
        Envía un prompt a Gemini y retorna la respuesta.

        Args:
            prompt (str): Prompt que describe el pseudocódigo o consulta a enviar.

        Returns:
            str: Respuesta generada por Gemini o un mensaje de error si la API falla.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error usando Gemini: {e}"

except Exception:
    # Fallback local: generar una respuesta heurística usando los analizadores locales
    """
    Fallback local para pruebas sin acceso a la API de Gemini.

    Este bloque utiliza los analizadores locales del proyecto para:
    - Analizar la complejidad del pseudocódigo.
    - Detectar patrones algorítmicos.
    - Realizar análisis línea por línea.
    """
    def ask_gemini(prompt: str) -> str:
        """Fallback simple para pruebas locales: analiza el pseudocódigo con los analizadores del proyecto

        Esto permite ejecutar tests sin acceso a la API de Gemini. La respuesta intenta imitar
        el formato esperado por `llm/validation.py`.
        """
        try:
            import re
            from core.complexity import ComplexityAnalyzer
            from core.patterns import PatternRecognizer

            # Extraer bloque de pseudocódigo si viene entre ``` ```
            code_match = re.search(r"```\s*(.*?)\s*```", prompt, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Intentar extraer tras la palabra 'Pseudocódigo:'
                idx = prompt.lower().find('pseudocódigo:')
                if idx != -1:
                    code = prompt[idx+len('pseudocódigo:'):].strip()
                else:
                    code = prompt

            analyzer = ComplexityAnalyzer()
            complexity = analyzer.analyze(code)

            recognizer = PatternRecognizer()
            patterns = recognizer.analyze(code)

            parts = []
            parts.append(f"Peor caso (Big-O): {complexity.big_o}")
            parts.append(f"Mejor caso (Omega): {complexity.omega}")
            parts.append(f"Caso promedio (Theta): {complexity.theta}")
            if complexity.recurrence:
                parts.append(f"Recurrencia: {complexity.recurrence}")

            if patterns:
                pat_names = [p.get('name') or p.get('description') for p in patterns]
                parts.append(f"Patrones detectados: {', '.join(pat_names)}")
            else:
                parts.append("Patrones detectados: Ninguno")

            # Línea por línea: usar el parser para obtener costos por línea (si está disponible)
            parts.append("Análisis línea por línea:")
            try:
                from core.parser import PseudocodeParser
                parser = PseudocodeParser()
                lines = parser.analyze_by_line(code)
                for li in lines:
                    if li.get('type') == 'summary':
                        continue
                    line_no = li.get('line')
                    time_cost = li.get('time_cost') or li.get('cost') or li.get('complexity') or ''
                    parts.append(f"Línea {line_no}: {time_cost}")
            except Exception:
                # Fallback simple si parser falla
                for i, line in enumerate(code.strip().split('\n'), start=1):
                    parts.append(f"Línea {i}: {line.strip()}")

            parts.append(f"Explicación: {complexity.explanation}")

            return "\n".join(parts)

        except Exception as e:
            return f"Fallback de Gemini falló: {e}"

    def fallback_ask_gemini(prompt: str) -> str:
        """
        Genera una respuesta heurística utilizando los analizadores locales.

        Args:
            prompt (str): Prompt que describe el pseudocódigo o consulta a analizar.

        Returns:
            str: Respuesta heurística que incluye análisis de complejidad, patrones detectados, 
                 y análisis línea por línea.
        """
        try:
            import re
            from core.complexity import ComplexityAnalyzer
            from core.patterns import PatternRecognizer

            # Extraer bloque de pseudocódigo si viene entre ``` ```
            code_match = re.search(r"```\s*(.*?)\s*```", prompt, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Intentar extraer tras la palabra 'Pseudocódigo:'
                idx = prompt.lower().find('pseudocódigo:')
                if idx != -1:
                    code = prompt[idx+len('pseudocódigo:'):].strip()
                else:
                    code = prompt

            analyzer = ComplexityAnalyzer()
            complexity = analyzer.analyze(code)

            recognizer = PatternRecognizer()
            patterns = recognizer.analyze(code)

            parts = []
            parts.append(f"Peor caso (Big-O): {complexity.big_o}")
            parts.append(f"Mejor caso (Omega): {complexity.omega}")
            parts.append(f"Caso promedio (Theta): {complexity.theta}")
            if complexity.recurrence:
                parts.append(f"Recurrencia: {complexity.recurrence}")

            if patterns:
                pat_names = [p.get('name') or p.get('description') for p in patterns]
                parts.append(f"Patrones detectados: {', '.join(pat_names)}")
            else:
                parts.append("Patrones detectados: Ninguno")

            # Línea por línea: usar el parser para obtener costos por línea (si está disponible)
            parts.append("Análisis línea por línea:")
            try:
                from core.parser import PseudocodeParser
                parser = PseudocodeParser()
                lines = parser.analyze_by_line(code)
                for li in lines:
                    if li.get('type') == 'summary':
                        continue
                    line_no = li.get('line')
                    time_cost = li.get('time_cost') or li.get('cost') or li.get('complexity') or ''
                    parts.append(f"Línea {line_no}: {time_cost}")
            except Exception:
                # Fallback simple si parser falla
                for i, line in enumerate(code.strip().split('\n'), start=1):
                    parts.append(f"Línea {i}: {line.strip()}")

            parts.append(f"Explicación: {complexity.explanation}")

            return "\n".join(parts)

        except Exception as e:
            return f"Fallback de Gemini falló: {e}"


# ============================================================================
# FUNCIONES CENTRALIZADAS PARA EXTRACCIÓN Y NORMALIZACIÓN DE COMPLEJIDADES
# ============================================================================

import re
from typing import Dict, Optional

def extract_gemini_complexities(gemini_response: str) -> Dict[str, Optional[str]]:
    """
    Extrae complejidades (Big-O, Omega, Theta) de la respuesta de Gemini.
    Función centralizada para uso consistente en todo el proyecto.
    
    MÁS ROBUSTA: Maneja múltiples formatos y responde a palabras clave.
    
    Args:
        gemini_response (str): Respuesta completa de Gemini
        
    Returns:
        Dict con claves 'big_o', 'omega', 'theta' y valores como "O(1)", "Ω(n)", etc.
    """
    complexities = {
        'big_o': None,
        'omega': None,
        'theta': None
    }
    
    text = gemini_response
    
    # ========== BUSCAR BIG-O ==========
    # Patrones flexibles para "peor caso"
    big_o_contexts = ['peor caso', 'big-o', 'worst case', 'upper bound']
    
    # Buscar en cada línea que contenga estas palabras clave
    for line in text.split('\n'):
        line_lower = line.lower()
        if any(context in line_lower for context in big_o_contexts):
            # Buscar O(...) en esta línea
            match = re.search(r'([oO]\([^)]+\))', line)
            if match:
                complexities['big_o'] = match.group(1)
                break
    
    # Si no encuentra, buscar el PRIMER O(...) del texto
    if not complexities['big_o']:
        match = re.search(r'([oO]\([^)]+\))', text)
        if match:
            complexities['big_o'] = match.group(1)
    
    # ========== BUSCAR OMEGA ==========
    # Patrones flexibles para "mejor caso"
    omega_contexts = ['mejor caso', 'omega', 'best case', 'lower bound']
    
    for line in text.split('\n'):
        line_lower = line.lower()
        if any(context in line_lower for context in omega_contexts):
            # Buscar Ω(...) o Omega(...) en esta línea
            match = re.search(r'((?:[ωΩ]|[Oo]mega)\([^)]+\))', line, re.IGNORECASE)
            if match:
                val = match.group(1)
                # Normalizar "omega" a símbolo Ω
                if val.lower().startswith('omega'):
                    val = 'Ω' + val[5:]
                complexities['omega'] = val
                break
    
    # Si no encuentra, buscar cualquier Ω o "omega"
    if not complexities['omega']:
        # Buscar símbolo Ω
        match = re.search(r'(Ω\([^)]+\))', text)
        if match:
            complexities['omega'] = match.group(1)
        else:
            # Buscar palabra "omega" (case insensitive)
            match = re.search(r'(omega\([^)]+\))', text, re.IGNORECASE)
            if match:
                val = match.group(1)
                if val.lower().startswith('omega'):
                    val = 'Ω' + val[5:]
                complexities['omega'] = val
    
    # ========== BUSCAR THETA ==========
    # Patrones flexibles para "caso promedio"
    theta_contexts = ['caso promedio', 'theta', 'average case', 'tight bound']
    
    for line in text.split('\n'):
        line_lower = line.lower()
        if any(context in line_lower for context in theta_contexts):
            # Buscar Θ(...) o Theta(...) en esta línea
            match = re.search(r'((?:[θΘ]|[Tt]heta)\([^)]+\))', line, re.IGNORECASE)
            if match:
                val = match.group(1)
                # Normalizar "theta" a símbolo Θ
                if val.lower().startswith('theta'):
                    val = 'Θ' + val[5:]
                complexities['theta'] = val
                break
    
    # Si no encuentra, buscar cualquier Θ o "theta"
    if not complexities['theta']:
        # Buscar símbolo Θ
        match = re.search(r'(Θ\([^)]+\))', text)
        if match:
            complexities['theta'] = match.group(1)
        else:
            # Buscar palabra "theta" (case insensitive)
            match = re.search(r'(theta\([^)]+\))', text, re.IGNORECASE)
            if match:
                val = match.group(1)
                if val.lower().startswith('theta'):
                    val = 'Θ' + val[5:]
                complexities['theta'] = val
    
    return complexities


def normalize_complexity_for_comparison(complexity: str) -> str:
    """
    Normaliza una complejidad para comparación consistente.
    IMPORTANTE: Preserva la función de complejidad completa, no solo el contenido.
    
    Args:
        complexity (str): Complejidad a normalizar (ej: "O(n)", "Ω(log n)", etc.)
        
    Returns:
        str: Versión normalizada conservando estructura
    """
    if not complexity:
        return ""
    
    # Paso 1: Limpiar espacios y convertir a minúsculas
    norm = complexity.strip().lower().replace(' ', '')
    
    # Paso 2: Normalizar símbolos griegos a 'o' para comparación
    # Preservamos pero normalizamos la notación
    norm = norm.replace('ω', 'o').replace('Ω', 'o')
    norm = norm.replace('θ', 'o').replace('Θ', 'o')
    norm = norm.replace('omega', 'o').replace('theta', 'o')
    
    # Paso 3: Normalizar el contenido entre paréntesis
    # Buscar O(...), o(...), etc.
    match = re.search(r'o\(([^)]+)\)', norm)
    if match:
        content = match.group(1)
        
        # Normalizar el contenido preservando estructura
        # Exponentes
        content = content.replace('^2', '²').replace('^3', '³')
        content = content.replace('**2', '²').replace('**3', '³')
        
        # Logaritmos
        content = content.replace('log(n)', 'logn')
        content = content.replace('log n', 'logn')
        content = content.replace('lg(n)', 'logn')
        content = content.replace('lg n', 'logn')
        
        # Multiplicaciones
        content = content.replace('*', '').replace('·', '').replace('×', '')
        
        # Retornar forma normalizada: O(contenido)
        return f'o({content})'
    
    # Fallback si no hay paréntesis
    return norm


def complexities_match(auto_complexity: str, gemini_complexity: Optional[str]) -> bool:
    """
    Verifica si dos complejidades son matemáticamente equivalentes.
    Maneja variaciones de notación y formato preservando precisión.
    
    Args:
        auto_complexity (str): Complejidad del análisis automático (ej: "O(2^n)")
        gemini_complexity (Optional[str]): Complejidad de Gemini (ej: "O(n)")
        
    Returns:
        bool: True si son equivalentes
    """
    if not gemini_complexity or not auto_complexity:
        return False
    
    # Normalizar ambas
    auto_norm = normalize_complexity_for_comparison(auto_complexity)
    gemini_norm = normalize_complexity_for_comparison(gemini_complexity)
    
    # Comparación directa
    if auto_norm == gemini_norm:
        return True
    
    # Extraer solo el contenido para comparación más flexible
    auto_match = re.search(r'o\(([^)]+)\)', auto_norm)
    gemini_match = re.search(r'o\(([^)]+)\)', gemini_norm)
    
    if not auto_match or not gemini_match:
        return False
    
    auto_content = auto_match.group(1)
    gemini_content = gemini_match.group(1)
    
    # Comparación directa del contenido
    if auto_content == gemini_content:
        return True
    
    # Equivalencias especiales conocidas
    equivalences = {
        'n2': ['n²', 'n*n', 'n^2'],
        'n3': ['n³', 'n*n*n', 'n^3'],
        'logn': ['log(n)', 'lgn', 'log n'],
        'nlogn': ['n*logn', 'nlog(n)', 'n*log(n)', 'n*log n'],
        '1': ['c', 'k', 'constant'],
        '2^n': ['2**n', 'exp(n)'],
    }
    
    # Normalizar contenido para comparación de equivalencias
    auto_clean = auto_content.replace('^', '').replace('**', '').replace('(', '').replace(')', '')
    gemini_clean = gemini_content.replace('^', '').replace('**', '').replace('(', '').replace(')', '')
    
    for standard, variants in equivalences.items():
        standard_clean = standard.replace('^', '').replace('**', '').replace('(', '').replace(')', '')
        
        auto_matches = auto_clean == standard_clean or auto_content in [standard] + variants
        gemini_matches = gemini_clean == standard_clean or gemini_content in [standard] + variants
        
        if auto_matches and gemini_matches:
            return True
    
    return False


def build_gemini_complexity_prompt(code: str) -> str:
    """
    Construye un prompt estándar y consistente para Gemini.
    Asegura que Gemini responda siempre en el formato esperado.
    
    Args:
        code (str): Pseudocódigo a analizar
        
    Returns:
        str: Prompt formateado para Gemini
    """
    prompt = f"""Analiza este pseudocódigo y proporciona EXACTAMENTE en este formato:

1. Peor caso (Big-O): O(...)
2. Mejor caso (Omega): Ω(...)
3. Caso promedio (Theta): Θ(...)
4. Recurrencia: T(n) = ... (solo si hay recursión)

PSEUDOCÓDIGO:
```
{code}
```

REGLAS IMPORTANTES:
- Usa notación estándar: O(1), O(n), O(n²), O(log n), O(n log n), O(2^n)
- Para Omega: Ω(1), Ω(n), etc. (o escribe Omega si no tienes el símbolo)
- Para Theta: Θ(1), Θ(n), etc. (o escribe Theta si no tienes el símbolo)
- Sé conciso, sin explicaciones largas
- Si Big-O = Omega, entonces Theta debe ser igual

EJEMPLO:
1. Peor caso (Big-O): O(n²)
2. Mejor caso (Omega): Ω(n)
3. Caso promedio (Theta): Θ(n²)
4. Recurrencia: No aplica"""
    
    return prompt
