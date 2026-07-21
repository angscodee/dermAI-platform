# 🔬 DermAI — Diagnóstico de Cáncer de Piel

Sistema de diagnóstico asistido por inteligencia artificial basado en imágenes
dermatoscópicas del dataset **ISIC 2019** (25,331 imágenes, clasificación binaria
benigno / maligno).

---

## Descripción

DermAI permite cargar una imagen dermatoscópica y obtener:

- Diagnóstico (benigno / maligno) con porcentaje de confianza
- Comparación simultánea entre cinco arquitecturas de deep learning
- Mapa de activación **Grad-CAM** para explicabilidad del diagnóstico
- Métricas de rendimiento: Accuracy, Sensitivity, Specificity, F1-Score, **MCC**
- Pruebas estadísticas de **McNemar** entre pares de modelos
- Generación de **reporte PDF** descargable

---

## Arquitectura de módulos

```
DermAI/
├── app.py                      ← Punto de entrada principal (Streamlit)
├── config.py                   ← Constantes globales y métricas de referencia
├── model_utils.py              ← Carga de modelos y funciones de predicción
├── preprocessing.py            ← Preprocesamiento de imágenes (resize, normalización)
├── activation_maps.py          ← Grad-CAM (explicabilidad)
├── hybrid_model_trainer.py     ← Entrenamiento asíncrono con queue thread-safe
├── hybrid_data_integrator.py   ← Integración de métricas de modelos híbridos
├── ui_components.py            ← Componentes de UI reutilizables (Streamlit)
├── data_processor.py           ← Comparación de modelos y cálculo de métricas
├── visualization.py            ← Gráficos: confusion matrix, dashboards, MCC, McNemar
├── metrics_calculator.py       ← MCC, test de McNemar (puro Python / scipy)
├── pdf_generator_optimized.py  ← Generación de reportes PDF (reportlab)
├── translations.py             ← Soporte multiidioma (es / en)
├── requirements.txt
├── README.md
└── tests/
    ├── test_data_processor.py
    └── test_metrics_calculator.py
```

Flujo de la aplicación dentro de `app.py`:

```
main()
  ├── initialize_page()            ← Streamlit page config
  ├── _load_models_cached()        ← Carga modelos .keras/.h5 con @st.cache_resource
  ├── _init_session_state()        ← Inicializa todas las claves de estado
  ├── update_model_metrics()       ← Integra métricas híbridas en session_state
  ├── setup_sidebar()              ← Config de usuario (modelo, idioma, umbral)
  ├── display_medical_disclaimer() ← ⚕️ Aviso médico al INICIO
  ├── _handle_hybrid_training()    ← Entrenamiento asíncrono con st.progress()
  ├── _handle_image_upload()       ← Carga + validación + preprocesamiento
  ├── _run_diagnosis()             ← Inferencia y resultados
  ├── _run_model_comparison()      ← Comparación multi-modelo
  ├── _run_metrics_section()       ← Confusion matrix + dashboard
  ├── _run_gradcam()               ← Mapa Grad-CAM
  ├── _run_statistical_section()   ← MCC + McNemar
  ├── _run_pdf_generation()        ← PDF descargable
  └── display_medical_disclaimer() ← ⚕️ Aviso médico al FINAL
```

---

## Instrucciones para ejecutar

### 1. Requisitos previos

- Python >= 3.11
- Modelos entrenados en `app/models/` (archivos `.keras` o `.h5`)

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la app

```bash
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`.

### 4. Ejecutar los tests

```bash
pytest tests/ -v
# Con cobertura:
pytest tests/ -v --cov=. --cov-report=term-missing
```

### 5. Entrenamiento en Google Colab

Si deseas entrenar los modelos reales desde cero con el dataset ISIC 2019 usando aceleración por GPU:
1. Sube el archivo [entrenamiento_dermai.ipynb](file:///c:/Users/ather/Downloads/P.U3/entrenamiento_dermai.ipynb) a **Google Colab**.
2. O bien, ejecuta el script [entrenamiento_colab.py](file:///c:/Users/ather/Downloads/P.U3/entrenamiento_colab.py) en tu entorno de Colab.
3. Descarga el archivo comprimido `modelos_entrenados_dermai.zip` generado al final del entrenamiento.
4. Descomprímelo y coloca los archivos `.keras` en la carpeta `app/models/` de este proyecto.

---

## Dataset

| Parámetro | Valor |
|-----------|-------|
| Nombre | ISIC 2019 (International Skin Imaging Collaboration) |
| Imágenes totales | 25,331 |
| Clases | Benigno / Maligno (binario) |
| Split train/val/test | 70% / 10% / 20% |
| Resolución entrada | 224 × 224 × 3 |

---

## Modelos disponibles

| Modelo | MCC | Accuracy | F1-Score |
|--------|-----|----------|----------|
| DenseNet121 | 0.764 | 88.2% | 87.9% |
| EfficientNetB0 | 0.748 | 87.4% | 87.1% |
| ResNet50 | 0.722 | 86.1% | 85.7% |
| InceptionV3 | 0.712 | 85.6% | 85.2% |
| MobileNetV2 | 0.696 | 84.8% | 84.1% |

---

## ⚕️ Aviso Médico — Disclaimer

> **Esta herramienta es únicamente para fines educativos y de investigación.**
>
> **No reemplaza el diagnóstico médico profesional.**
> Consulte siempre a un dermatólogo certificado para el diagnóstico y tratamiento
> de lesiones cutáneas. Los resultados de esta IA deben considerarse como una
> herramienta de apoyo, **no como un diagnóstico definitivo**.
>
> Los desarrolladores no asumen responsabilidad por decisiones clínicas tomadas
> con base en los resultados de este sistema.

---

## Licencia

MIT License — uso libre para fines educativos y de investigación.
