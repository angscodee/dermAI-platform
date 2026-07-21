# backend/app/api/chatbot.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot"])

class ChatMessage(BaseModel):
    message: str
    language: str = "es"

@router.post("/query")
def process_chat_message(msg: ChatMessage):
    user_msg = msg.message.lower()
    lang = msg.language
    
    if "hola" in user_msg or "hello" in user_msg or "hi" in user_msg:
        reply = "¡Hola! Soy el asistente inteligente de DermAI. ¿En qué puedo ayudarte hoy?" if lang == "es" else "Hello! I am the DermAI intelligent assistant. How can I help you today?"
    elif "modelo" in user_msg or "model" in user_msg:
        reply = "Contamos con 3 modelos clásicos (ResNet50, EfficientNetB0, MobileNetV2) y 2 modelos híbridos (DenseNet121+Attention, EfficientNet+SpatialFusion)." if lang == "es" else "We feature 3 classic models (ResNet50, EfficientNetB0, MobileNetV2) and 2 hybrid models (DenseNet121+Attention, EfficientNet+SpatialFusion)."
    elif "estadística" in user_msg or "prueba" in user_msg or "stat" in user_msg:
        reply = "Implementamos 5 pruebas robustas: Kolmogorov-Smirnov en logit gaps, Mann-Whitney U, Morgan-Pitman para varianzas de error, McNemar y Multiplicadores de Lagrange (LM)." if lang == "es" else "We implement 5 robust tests: Kolmogorov-Smirnov on logit gaps, Mann-Whitney U, Morgan-Pitman for error variance, McNemar, and Lagrange Multipliers (LM)."
    elif "reporte" in user_msg or "report" in user_msg:
        reply = "Puedes exportar los resultados completos en formatos PDF, Word (.docx) y Excel (.xlsx) desde la pestaña de Reportes." if lang == "es" else "You can export the full results in PDF, Word (.docx), and Excel (.xlsx) formats from the Reports tab."
    else:
        reply = f"He procesado tu consulta: '{msg.message}'. Puedes pedirme información sobre EDA, modelos híbridos, pruebas estadísticas o generación de reportes." if lang == "es" else f"I processed your query: '{msg.message}'. You can ask me about EDA, hybrid models, statistical tests, or report generation."
        
    return {
        "reply": reply,
        "language": lang
    }
