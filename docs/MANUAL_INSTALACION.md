# 🛠️ Manual de Instalación — Plataforma DermAI Full-Stack

Este manual describe el procedimiento para desplegar la plataforma localmente y en servidores de producción (Render / Vercel / GitHub).

---

## 1. Requisitos Previos

* **Python:** 3.11 o superior
* **Node.js:** v18.0.0 o superior
* **Git:** Para clonar el repositorio de GitHub

---

## 2. Instalación y Despliegue del Backend (FastAPI)

1. Abre una terminal y navega a la carpeta del backend:
   ```bash
   cd backend
   ```

2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

3. Inicia el servidor de FastAPI:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   * La documentación interactiva Swagger estará disponible en: `http://localhost:8000/docs`.

---

## 3. Instalación y Despliegue del Frontend (Next.js)

1. Abre una nueva terminal y navega a la carpeta del frontend:
   ```bash
   cd frontend
   ```

2. Instala los paquetes de Node:
   ```bash
   npm install
   ```

3. Ejecutar el servidor de desarrollo:
   ```bash
   npm run dev
   ```
   * La aplicación web estará disponible en: `http://localhost:3000`.

---

## 4. Despliegue en la Nube y GitHub

1. **GitHub:** Subir los cambios incluyendo ambas carpetas (`backend/` y `frontend/`).
2. **Frontend:** Conectar el repositorio de GitHub con **Vercel** para despliegue automatizado.
3. **Backend:** Desplegar la carpeta `backend/` en **Render.com** o **Hugging Face Spaces**.
