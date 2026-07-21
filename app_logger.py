"""
app_logger.py
Logging estructurado para DermAI. (Prioridad Alta #4)

Configura un logger con:
  - Archivo: app.log (nivel DEBUG en adelante)
  - Consola: nivel WARNING en adelante
  - Formato: timestamp | nivel | módulo | mensaje
"""

import logging
import logging.handlers
import os
from pathlib import Path

LOG_FILE = Path(__file__).parent / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Evitar añadir handlers duplicados si el módulo se importa varias veces
_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger con el nombre dado.
    El primer llamado configura los handlers globales.

    Args:
        name: nombre del módulo, ej. 'model_utils', 'app'

    Returns:
        logging.Logger listo para usar.
    """
    global _configured
    if not _configured:
        _setup_root_logger()
        _configured = True
    return logging.getLogger(name)


def _setup_root_logger():
    """Configura handlers del logger raíz una sola vez."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # --- Handler de archivo (rotante, máx 5 MB, 3 backups) ---
    try:
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(fh)
    except OSError as e:
        # Si no se puede escribir el log (permisos, etc.) seguir sin crashear
        print(f"[app_logger] No se pudo crear app.log: {e}")

    # --- Handler de consola (solo WARNING+) ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(ch)
