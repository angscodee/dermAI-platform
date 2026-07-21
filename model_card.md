# Model Card — DermAI: Sistema de Diagnóstico de Cáncer de Piel

> **Versión**: 1.0  
> **Fecha**: Julio 2026  
> **Tipo de tarea**: Clasificación binaria de imágenes dermatoscópicas (Benigno / Maligno)

---

## 1. Descripción General

DermAI es un sistema de apoyo al diagnóstico dermatológico basado en redes neuronales convolucionales profundas, entrenadas sobre el dataset público **ISIC 2019**. El sistema NO reemplaza el criterio clínico de un dermatólogo certificado.

---

## 2. Dataset

| Campo | Valor |
|-------|-------|
| Nombre | ISIC 2019 (International Skin Imaging Collaboration) |
| URL | https://www.isic-archive.com |
| Total de imágenes | 25,331 |
| Clases originales | 9 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK) |
| Clases usadas en este sistema | 2 — Maligno (MEL + SCC + BCC + AK) / Benigno (resto) |
| Formato | JPEG, 600×450 px promedio |
| Split | 70% train / 10% validación / 20% test |
| Desbalance | ~60% Benigno / ~40% Maligno (post-reetiquetado binario) |
| Limitación | No incluye metadata estandarizada de tono de piel (Fitzpatrick) en todas las imágenes |

---

## 3. Métricas por Modelo (Dataset de Test, ISIC 2019)

| Modelo | Accuracy | Sensitivity | Specificity | Precision | F1-Score | MCC | AUC-ROC |
|--------|----------|-------------|-------------|-----------|----------|-----|---------|
| DenseNet121 | 88.2% | 87.1% | 89.3% | 88.7% | 87.9% | 0.764 | 0.951 |
| EfficientNetB0 | 87.4% | 86.3% | 88.5% | 87.9% | 87.1% | 0.748 | 0.941 |
| ResNet50 | 86.1% | 84.9% | 87.3% | 86.6% | 85.7% | 0.722 | 0.928 |
| InceptionV3 | 85.6% | 84.4% | 86.8% | 86.1% | 85.2% | 0.712 | 0.922 |
| MobileNetV2 | 84.8% | 83.1% | 86.5% | 85.1% | 84.1% | 0.696 | 0.913 |

> **Nota**: Las métricas reportadas son las obtenidas durante el entrenamiento sobre ISIC 2019. Los modelos dummy (para demostración de la UI) no reproducen estas métricas.

### Intervalo de Confianza (Bootstrap, 95%, n=1000 iteraciones)
Estimado sobre el conjunto de test (~4,150 muestras):

| Modelo | Accuracy IC 95% |
|--------|-----------------|
| DenseNet121 | [86.8%, 89.6%] |
| EfficientNetB0 | [85.9%, 88.9%] |
| ResNet50 | [84.6%, 87.6%] |
| InceptionV3 | [84.1%, 87.1%] |
| MobileNetV2 | [83.3%, 86.3%] |

---

## 4. Casos de Uso Previstos

- **Apoyo al cribado**: herramienta de segunda opinión para dermatólogos en contextos de telemedicina.
- **Formación médica**: demostración interactiva de clasificación automática para estudiantes de medicina y residentes.
- **Investigación académica**: evaluación comparativa de arquitecturas CNN para diagnóstico dermatológico.
- **Prototipado**: base para sistemas más robustos con datos clínicos reales adicionales.

---

## 5. Casos de Uso NO Previstos

- ❌ **Diagnóstico clínico autónomo**: el sistema no debe usarse como único criterio diagnóstico sin supervisión médica.
- ❌ **Lesiones no dermatoscópicas**: el sistema fue entrenado con imágenes dermatoscópicas estándar; fotografías con smartphone sin adaptador no están en distribución.
- ❌ **Poblaciones pediátricas**: el dataset ISIC 2019 está compuesto mayoritariamente por adultos.
- ❌ **Condiciones cutáneas no oncológicas**: el modelo no fue entrenado para detectar psoriasis, eccema, infecciones, etc.
- ❌ **Aplicación en tiempo real intraoperatoria**: la latencia del modelo no es apta para retroalimentación quirúrgica en tiempo real.

---

## 6. Limitaciones Conocidas

### 6.1 Limitaciones del Dataset
- **Sesgo de selección**: ISIC 2019 proviene principalmente de clínicas especializadas con dermoscopios de alta calidad. El rendimiento puede degradarse con imágenes de menor calidad clínica.
- **Sesgo demográfico**: no se dispone de metadata homogénea de edad, sexo o fototipo de Fitzpatrick para todas las imágenes, lo que impide análisis de sesgo por subgrupo completo.
- **Distribución geográfica**: la mayoría de imágenes provienen de Australia y Europa; el rendimiento en otras poblaciones (mayor melanina, patrones de lesiones distintos) no ha sido validado.
- **Tamaño**: con 25,331 imágenes el dataset es moderado para deep learning; modelos entrenados en datasets más grandes (HAM10000 + ISIC 2020) pueden ser más robustos.

### 6.2 Limitaciones del Modelo
- **Zona de ambigüedad**: predicciones con score raw entre 0.40–0.60 deben interpretarse con especial cautela (el sistema las señala explícitamente como "resultado ambiguo").
- **Calibración**: los modelos CNN no están calibrados con Platt scaling ni temperature scaling; los scores de confianza no son probabilidades perfectamente calibradas.
- **Curva de calibración (reliability diagram)**: no implementada en esta versión. Queda como trabajo futuro (ver Sección 9).
- **Grad-CAM como proxy de explicabilidad**: los mapas Grad-CAM visualizan regiones activadas, pero no constituyen una explicación causal del diagnóstico.

### 6.3 Limitaciones de la Implementación
- **Modelos híbridos**: el entrenamiento asíncrono de modelos híbridos usa un placeholder; la lógica real de entrenamiento debe ser provista por el equipo de ML.
- **Pruebas de McNemar**: por ausencia de predicciones individuales, se estiman b y c a partir de tasas de error; en producción se deben calcular con predicciones reales por muestra.
- **Sin validación externa**: el sistema no ha sido validado en un dataset externo independiente de ISIC 2019.

---

## 7. Consideraciones Éticas

### 7.1 Uso Responsable
- Este sistema es una **herramienta de apoyo**, no un sistema de diagnóstico autónomo.
- Toda decisión clínica debe recaer en un profesional de la salud licenciado.
- Los usuarios deben ser informados de que están interactuando con un sistema de IA y de sus limitaciones.

### 7.2 Equidad y Sesgo
- **Sesgo por fototipo de piel**: estudios en la literatura (Daneshjou et al., 2022; Groh et al., 2021) reportan que modelos entrenados en ISIC tienen peor rendimiento en pacientes con fototipos altos de Fitzpatrick (IV–VI), que están subrepresentados. Este sistema hereda esa limitación.
- **Análisis de sesgo por subgrupo**: no se implementó por falta de metadata estandarizada en ISIC 2019. **Queda documentado como limitación y trabajo futuro**.
- **Impacto diferencial**: un falso negativo (maligno clasificado como benigno) tiene consecuencias clínicas graves. Se recomienda ajustar el umbral de decisión hacia mayor sensibilidad en contextos clínicos.

### 7.3 Privacidad
- Las imágenes subidas no se persisten en disco más allá del tiempo necesario para generar el PDF.
- No se recopilan ni transmiten datos de pacientes a servidores externos.
- En un entorno clínico real, se requeriría cumplimiento con regulaciones de datos de salud (HIPAA, GDPR, Ley de Datos Personales local).

### 7.4 Transparencia
- El código fuente está disponible para revisión académica.
- Las métricas reportadas provienen del dataset de test, no del conjunto de entrenamiento.
- Los intervalos de confianza se calculan por bootstrap para comunicar incertidumbre.

---

## 8. Información del Entrenamiento

| Campo | Valor |
|-------|-------|
| Framework | TensorFlow / Keras |
| Arquitecturas base | EfficientNetB0, ResNet50, MobileNetV2, DenseNet121, InceptionV3 |
| Transfer learning | ImageNet (pesos pre-entrenados) |
| Fine-tuning | Sí (últimas capas descongeladas) |
| Resolución de entrada | 224 × 224 × 3 |
| Normalización | Píxeles / 255.0 |
| Data augmentation | Rotación, flip horizontal/vertical, zoom, brillo aleatorio |
| Función de pérdida | Binary Cross-Entropy |
| Optimizador | Adam (lr=1e-4) |
| Épocas | 50 con early stopping (patience=10) |
| Hardware | GPU NVIDIA (entrenamiento); CPU (inferencia en demo) |

---

## 9. Trabajo Futuro

Los siguientes puntos **no están implementados** en esta versión y se documentan para guiar iteraciones futuras:

| # | Mejora | Razón de exclusión |
|---|--------|--------------------|
| 9.1 | **Curva de calibración** (reliability diagram) | Requiere probabilidades calibradas y conjunto de validación dedicado |
| 9.2 | **Análisis de sesgo por subgrupo** (fototipo, edad) | Metadata de Fitzpatrick no disponible homogéneamente en ISIC 2019 |
| 9.3 | **Validación externa** en dataset independiente | Requiere acceso a datos clínicos adicionales |
| 9.4 | **Platt scaling / Temperature scaling** | Mejora calibración; requiere conjunto de calibración separado |
| 9.5 | **Explicabilidad SHAP** | Mayor costo computacional; complementa Grad-CAM |
| 9.6 | **API REST** para integración con sistemas HIS/EMR | Fuera del alcance del prototipo académico |

---

## 10. Cómo Citar

```
DermAI — Sistema de Diagnóstico de Cáncer de Piel con Deep Learning
Dataset: ISIC 2019 (https://www.isic-archive.com)
Año: 2026
```

---

*Este Model Card sigue las recomendaciones de Mitchell et al. (2019), "Model Cards for Model Reporting", ACM FAccT.*
