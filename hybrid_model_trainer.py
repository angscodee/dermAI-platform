"""
hybrid_model_trainer.py
Entrenamiento asíncrono de modelos híbridos con seguimiento de progreso thread-safe.

Fix #2: usa queue.Queue para comunicar el progreso desde el hilo de entrenamiento
al hilo principal de Streamlit, evitando acceso concurrente a st.session_state.
"""

import threading
import queue
import os
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Estado compartido (thread-safe)
# ---------------------------------------------------------------------------
_progress_queue: queue.Queue = queue.Queue()

# Directorio donde se guardan los modelos híbridos
HYBRID_MODELS_DIR = os.path.join(os.path.dirname(__file__), "app", "models", "hybrid")

_training_thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def check_hybrid_models_exist() -> tuple:
    """
    Verifica si existen modelos híbridos entrenados en disco.

    Returns:
        (exists: bool, model_names: list[str])
    """
    if not os.path.isdir(HYBRID_MODELS_DIR):
        return False, []

    hybrid_files = [
        f for f in os.listdir(HYBRID_MODELS_DIR)
        if f.endswith((".keras", ".h5"))
    ]
    names = [os.path.splitext(f)[0] for f in hybrid_files]
    return bool(names), names


def train_hybrid_models_async(progress_callback=None):
    """
    Inicia el entrenamiento de modelos híbridos en un hilo de fondo.
    El progreso se comunica a través de _progress_queue (thread-safe).

    El parámetro `progress_callback` se mantiene por compatibilidad con el
    código existente, pero ya NO se llama directamente desde el hilo worker.
    En su lugar, los mensajes se envían a la queue y se leen con
    get_training_progress() desde el hilo principal.
    """
    global _training_thread

    if _training_thread is not None and _training_thread.is_alive():
        return  # ya está corriendo

    def _worker():
        try:
            _put(0.0, "Iniciando entrenamiento de modelos híbridos…")
            # --- Aquí iría el entrenamiento real ---
            # Simulación con pasos incrementales:
            steps = 10
            for i in range(1, steps + 1):
                time.sleep(0.5)          # placeholder para trabajo real
                progress = i / steps
                _put(progress, f"Época {i}/{steps} completada")

            _put(1.0, "✅ Entrenamiento finalizado")
        except Exception as e:
            _put(-1.0, f"❌ Error durante el entrenamiento: {e}")

    _training_thread = threading.Thread(target=_worker, daemon=True)
    _training_thread.start()


def get_training_progress() -> Optional[tuple]:
    """
    Lee el último mensaje de progreso disponible en la queue.
    Debe llamarse desde el hilo principal de Streamlit.

    Returns:
        (progress: float, message: str)
          progress == -1.0  → error
          progress == 1.0   → completado
          0 <= progress < 1 → en curso
          None              → sin mensajes nuevos
    """
    last = None
    try:
        # Drena todos los mensajes pendientes y queda con el más reciente
        while True:
            last = _progress_queue.get_nowait()
    except queue.Empty:
        pass
    return last  # None si no había mensajes


def is_training_active() -> bool:
    """True si el hilo de entrenamiento está corriendo."""
    return _training_thread is not None and _training_thread.is_alive()


# ---------------------------------------------------------------------------
# Helper privado
# ---------------------------------------------------------------------------
def _put(progress: float, message: str):
    """Envía un mensaje de progreso a la queue."""
    _progress_queue.put((progress, message))
