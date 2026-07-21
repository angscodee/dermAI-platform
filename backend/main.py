# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, eda, training, cross_validation, tuning, statistical, reports, chatbot, predict, ensemble

app = FastAPI(
    title="DermAI Platform API",
    description="Backend en FastAPI para Diagnóstico por Redes Neuronales Clásicas e Híbridas, Pruebas Estadísticas Robustas y Reportes",
    version="2.0.0"
)

# Configuración de CORS para permitir solicitudes desde el Frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de Routers
app.include_router(auth.router)
app.include_router(eda.router)
app.include_router(training.router)
app.include_router(cross_validation.router)
app.include_router(tuning.router)
app.include_router(statistical.router)
app.include_router(reports.router)
app.include_router(chatbot.router)
app.include_router(predict.router)
app.include_router(ensemble.router)

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "DermAI FastAPI Service Running",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
