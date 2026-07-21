# 📘 Manual de Usuario — Plataforma DermAI Full-Stack

Este manual guía a los investigadores, médicos y evaluadores a través de todas las funcionalidades del sistema.

---

## 1. Acceso y Autenticación

1. Abre `http://localhost:3000` en tu navegador.
2. Ingresa las credenciales por defecto:
   * **Usuario:** `admin`
   * **Contraseña:** `admin123`
3. Haz clic en **Ingresar al Sistema** para acceder al Dashboard interactivo.

---

## 2. Controles Globales (Header)

* **Selector Bilingüe (ES / EN):** Cambia instantáneamente el idioma de toda la interfaz entre Español e Inglés.
* **Conmutador Modo Claro / Oscuro:** Alterna el tema visual de Tailwind CSS entre claro u oscuro.

---

## 3. Módulos del Dashboard

1. **Pestaña 1 — EDA:** Muestra los estadísticos descriptivos, total de muestras y distribución de clases.
2. **Pestaña 2 — Entrenamiento:** Presiona **Iniciar Entrenamiento** para entrenar las 3 arquitecturas clásicas (ResNet50, EfficientNetB0, MobileNetV2) y las 2 híbridas (DenseNet121+Attention, EfficientNet+SpatialFusion). Al finalizar, el mejor modelo se guarda automáticamente como `.keras`.
3. **Pestaña 3 — Validación Cruzada:** Ejecuta el 5-Fold Stratified Cross Validation.
4. **Pestaña 4 — Hiperparámetros:** Muestra el ranking de combinaciones de hiperparámetros (Learning Rate, Dropout, Optimizador).
5. **Pestaña 5 — Pruebas Estadísticas:** Ejecuta las 5 pruebas robustas (KS logit gaps, Mann-Whitney U, Morgan-Pitman, McNemar y LM).
6. **Pestaña 6 — Reportes:** Descarga el informe consolidado en formatos **PDF**, **Word (.docx)** y **Excel (.xlsx)**.

---

## 4. Chatbot de Asistencia por Voz y Texto

* Puedes escribir en el campo de texto o presionar el botón de **🎙️ Hablar** para formular preguntas por voz a través de tu micrófono.
* El asistente te responderá tanto en texto como con lectura en voz alta (Speech Synthesis).
