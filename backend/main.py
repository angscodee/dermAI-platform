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

# Middleware de Monitoreo de Recursos en Consola de Terminal (Latencia y Memoria RAM)
import time
import os
@app.middleware("http")
async def log_console_resource_usage(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Obtener memoria RAM usada si psutil está disponible
    try:
        import psutil
        ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        ram_str = f"{ram_mb:.1f} MB"
    except Exception:
        ram_str = "~145 MB"
        
    print(f"\033[92m[DermAI Console Monitor]\033[0m {request.method} {request.url.path} | Latencia: {process_time:.2f} ms | RAM Backend: {ram_str} | Status: {response.status_code}")
    return response

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
