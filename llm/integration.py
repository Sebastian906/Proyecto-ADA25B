"""
Módulo `integration.py`

Este módulo se encarga de la integración con la API de Gemini para enviar prompts y recibir respuestas. 
Incluye un fallback local que utiliza los analizadores del proyecto para generar respuestas heurísticas 
cuando la API no está disponible.

Funciones principales:
    - ask_gemini: Envía un prompt a Gemini y retorna la respuesta.
    - fallback_ask_gemini: Genera una respuesta heurística utilizando los analizadores locales.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _fallback_local_analysis(prompt: str) -> str:
    """
    Genera una respuesta heurística utilizando los analizadores locales.
    
    Args:
        prompt (str): Prompt que describe el pseudocódigo o consulta a analizar.
    
    Returns:
        str: Respuesta heurística en formato estructurado.
    """
    try:
        import re
        from core.complexity import ComplexityAnalyzer
        from core.patterns import PatternRecognizer

        # Extraer bloque de pseudocódigo
        code_match = re.search(r"```\s*(.*?)\s*```", prompt, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
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
        parts.append("="*80)
        parts.append("ANÁLISIS AUTOMÁTICO (Fallback Local)")
        parts.append("="*80)
        parts.append("")
        parts.append(f"1. Peor caso (Big-O): {complexity.big_o}")
        parts.append(f"2. Mejor caso (Omega): {complexity.omega}")
        parts.append(f"3. Caso promedio (Theta): {complexity.theta}")
        
        if complexity.recurrence and complexity.recurrence != "No detectada":
            parts.append(f"4. Recurrencia (si aplica): {complexity.recurrence}")
        else:
            parts.append("4. Recurrencia (si aplica): No aplica")
        
        parts.append("")

        if patterns:
            pat_names = [p.get('name') or p.get('description') for p in patterns]
            parts.append(f"Patrones detectados: {', '.join(pat_names)}")
        else:
            parts.append("Patrones detectados: Ninguno")

        parts.append("")
        parts.append("Análisis línea por línea:")
        try:
            from core.parser import PseudocodeParser
            parser = PseudocodeParser()
            lines = parser.analyze_by_line(code)
            for li in lines:
                if li.get('type') == 'summary':
                    continue
                line_no = li.get('line')
                time_cost = li.get('time_cost') or li.get('cost') or li.get('complexity') or 'O(1)'
                parts.append(f"Línea {line_no}: {time_cost}")
        except Exception:
            for i, line in enumerate(code.strip().split('\n'), start=1):
                if line.strip():
                    parts.append(f"Línea {i}: O(1)")

        parts.append("")
        parts.append(f"Explicación: {complexity.explanation}")
        parts.append("")
        parts.append("="*80)

        return "\n".join(parts)

    except Exception as e:
        return f"Error en análisis local: {e}"

try:
    import google.generativeai as genai
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
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            generation_config = {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config 
            )
            return response.text
        except Exception as e:
            print(f"Gemini API falló (usando fallback local): {str(e)[:100]}...")
            return _fallback_local_analysis(prompt)

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
                for i, line in enumerate(code.strip().split('\n'), start=1):
                    parts.append(f"Línea {i}: {line.strip()}")

            parts.append(f"Explicación: {complexity.explanation}")

            return "\n".join(parts)

        except Exception as e:
            return f"Fallback de Gemini falló: {e}"

    def fallback_ask_gemini(prompt: str) -> str:
        """
        Genera una respuesta heurística utilizando los analizadores locales.
        """
        return ask_gemini(prompt)