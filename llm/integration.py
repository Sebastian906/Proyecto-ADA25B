import os
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

# Configurar API key de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Función para enviar prompts a Gemini
def ask_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error usando Gemini: {e}"
