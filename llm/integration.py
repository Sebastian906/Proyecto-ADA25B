import os
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
        """Enviar prompt a Gemini (biblioteca oficial)."""
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error usando Gemini: {e}"

except Exception:
    # Fallback local: generar una respuesta heurística usando los analizadores locales
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
