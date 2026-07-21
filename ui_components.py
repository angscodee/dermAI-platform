"""
ui_components.py
Componentes de UI reutilizables para la aplicación Streamlit.

Mejoras v2:
  - display_ambiguous_result(): alerta para predicciones cercanas al umbral (#1)
  - display_model_limitations(): sección de limitaciones en la app (#3)
  - add_to_diagnosis_history() / display_diagnosis_history(): historial de sesión (#7)
"""

import streamlit as st
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Configuración de la barra lateral
# ---------------------------------------------------------------------------
def setup_sidebar(models: dict, t: dict, available_languages: dict) -> dict:
    """Configura la barra lateral y retorna la configuración del usuario."""
    with st.sidebar:
        st.title("⚙️ Configuración")

        # Idioma
        lang = st.selectbox(
            "🌐 Idioma / Language",
            options=list(available_languages.keys()),
            index=0,
        )

        st.markdown("---")

        # Modelo
        model_names = list(models.keys())
        selected_model = st.selectbox(
            t.get("select_model", "🤖 Modelo de Diagnóstico"),
            options=model_names,
            index=0,
        )

        st.markdown("---")

        # Umbral de decisión
        decision_threshold = st.slider(
            t.get("decision_threshold", "⚖️ Umbral de Decisión"),
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help=t.get("threshold_help", "Valores > umbral → Maligno"),
        )

        # Umbral de confianza para interpretación
        confidence_threshold = st.slider(
            t.get("confidence_threshold", "🎯 Umbral de Confianza"),
            min_value=50,
            max_value=99,
            value=70,
            step=5,
        )

        st.markdown("---")

        # Modo debug
        debug_mode = st.checkbox(
            t.get("debug_mode", "🔧 Modo Debug"),
            value=False,
        )

        # Grad-CAM
        show_gradcam = st.checkbox(
            t.get("show_gradcam", "🔥 Mostrar Grad-CAM"),
            value=True,
            help=t.get("gradcam_help", "Mapa de activación para explicabilidad"),
        )

        # Entrenamiento híbrido
        st.markdown("---")
        st.subheader("🧬 Modelos Híbridos")
        if st.button(t.get("train_hybrid", "Entrenar Modelos Híbridos"), use_container_width=True):
            st.session_state["start_hybrid_training"] = True

    return {
        "lang": lang,
        "selected_model": selected_model,
        "decision_threshold": decision_threshold,
        "confidence_threshold": confidence_threshold,
        "debug_mode": debug_mode,
        "show_gradcam": show_gradcam,
    }


# ---------------------------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------------------------
def display_main_header(t: dict):
    st.title("🔬 DermAI — Diagnóstico de Cáncer de Piel")
    st.markdown(
        t.get(
            "app_description",
            "Sistema de diagnóstico asistido por inteligencia artificial basado en el dataset ISIC 2019.",
        )
    )


# ---------------------------------------------------------------------------
# Sección de carga de imagen
# ---------------------------------------------------------------------------
def display_image_upload_section(t: dict):
    """
    Muestra el uploader de imagen con validación de tipo y tamaño (Fix #9).

    Returns:
        El objeto UploadedFile válido, o None.
    """
    st.subheader("📤 " + t.get("upload_title", "Cargar Imagen Dermatoscópica"))
    st.markdown(t.get("upload_instructions", "Sube una imagen de la lesión cutánea para su análisis."))

    uploaded_file = st.file_uploader(
        t.get("upload_label", "Selecciona una imagen (.jpg, .jpeg o .png)"),
        type=["jpg", "jpeg", "png"],
        help=t.get("upload_help", "Formatos soportados: JPG, JPEG, PNG. Máximo 10 MB."),
    )

    if uploaded_file is not None:
        # Validación de tipo de archivo
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png"):
            st.error(t.get("upload_invalid_type", "Tipo de archivo no permitido. Usa .jpg, .jpeg o .png."))
            return None

        # Validación de tamaño (10 MB)
        MAX_SIZE_BYTES = 10 * 1024 * 1024
        if uploaded_file.size > MAX_SIZE_BYTES:
            st.error(t.get("upload_too_large", "El archivo supera el límite de 10 MB. Por favor, reduce el tamaño."))
            return None

    return uploaded_file


# ---------------------------------------------------------------------------
# Comparación de imágenes
# ---------------------------------------------------------------------------
def display_image_comparison(original_image, processed_image: np.ndarray, t: dict):
    """Muestra la imagen original y la preprocesada en dos columnas."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🖼️ Imagen Original**")
        st.image(original_image, use_container_width=True)
    with col2:
        st.markdown("**⚙️ Imagen Preprocesada (224×224)**")
        # Remover dimensión de batch para mostrar
        disp = processed_image[0] if processed_image.ndim == 4 else processed_image
        st.image(disp, use_container_width=True, clamp=True)


# ---------------------------------------------------------------------------
# Resultados del diagnóstico
# ---------------------------------------------------------------------------
def display_diagnosis_results(diagnosis: str, confidence_percent: float,
                               raw_confidence: float, t: dict):
    col1, col2, col3 = st.columns(3)
    with col1:
        if diagnosis == "Maligno":
            st.error(t.get("diagnosis_malignant", "⚠️ POSIBLE LESIÓN MALIGNA"))
        else:
            st.success(t.get("diagnosis_benign", "✅ LESIÓN PROBABLEMENTE BENIGNA"))
    with col2:
        st.metric(label="Confianza", value=f"{confidence_percent:.1f}%")
    with col3:
        st.metric(label="Score Raw", value=f"{raw_confidence:.4f}")


def display_debug_info(processed_image: np.ndarray, model, threshold: float, t: dict):
    """Muestra información de debug del modelo y la imagen."""
    with st.expander("🔧 Información de Debug", expanded=False):
        st.write(f"- **Shape de entrada:** {processed_image.shape}")
        st.write(f"- **Valor mín/máx:** {processed_image.min():.3f} / {processed_image.max():.3f}")
        st.write(f"- **Umbral aplicado:** {threshold}")
        try:
            st.write(f"- **Input shape del modelo:** {model.input_shape}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Interpretación
# ---------------------------------------------------------------------------
def display_interpretation(confidence_percent: float, confidence_threshold: int,
                            diagnosis: str, t: dict):
    st.markdown("---")
    st.subheader("📋 Interpretación")
    if confidence_percent >= confidence_threshold:
        level = "Alta" if confidence_percent >= 85 else "Moderada"
        st.info(f"**Confianza {level}** ({confidence_percent:.1f}%) — Diagnóstico: {diagnosis}")
    else:
        st.warning(
            f"**Confianza Baja** ({confidence_percent:.1f}%) — "
            "Se recomienda revisión por especialista."
        )


# ---------------------------------------------------------------------------
# Tabla de comparación de modelos
# ---------------------------------------------------------------------------
def display_model_comparison_table(df_comparison):
    st.dataframe(df_comparison, use_container_width=True)


def display_consistency_analysis(comparison_results: list[dict], t: dict):
    from data_processor import analyze_consistency
    analysis = analyze_consistency(comparison_results)
    if analysis["consistent"]:
        st.success(
            f"✅ **Diagnóstico consistente** entre modelos "
            f"({analysis['agreement_rate']}% de acuerdo) — {analysis['majority_diagnosis']}"
        )
    else:
        st.warning(
            f"⚠️ **Diagnóstico inconsistente** entre modelos "
            f"({analysis['agreement_rate']}% de acuerdo)"
        )


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def display_metrics_explanation(t: dict):
    with st.expander("ℹ️ ¿Qué significan estas métricas?", expanded=False):
        st.markdown("""
        - **Accuracy**: porcentaje de predicciones correctas sobre el total.
        - **Sensitivity (Recall)**: tasa de detección de lesiones malignas reales.
        - **Specificity**: tasa de correcta identificación de lesiones benignas.
        - **Precision**: de los casos marcados como malignos, cuántos lo son realmente.
        - **F1-Score**: media armónica de precision y sensitivity.
        - **MCC**: coeficiente balanceado para clases desequilibradas (−1 a +1).
        - **AUC-ROC**: área bajo la curva ROC; 1.0 = modelo perfecto.
        """)


def display_metrics_in_columns(metrics: dict, t: dict):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy",    f"{metrics.get('accuracy', 0)*100:.1f}%")
        st.metric("Sensitivity", f"{metrics.get('sensitivity', 0)*100:.1f}%")
        st.metric("Specificity", f"{metrics.get('specificity', 0)*100:.1f}%")
    with col2:
        st.metric("Precision",   f"{metrics.get('precision', 0)*100:.1f}%")
        st.metric("F1-Score",    f"{metrics.get('f1_score', 0)*100:.1f}%")
        st.metric("MCC",         f"{metrics.get('mcc', 0):.3f}")


def display_mcc_interpretation(mcc_data: list[dict], t: dict):
    if not mcc_data:
        return
    best = max(mcc_data, key=lambda x: x["mcc"])
    st.markdown(f"**Mejor modelo por MCC:** {best['model']} ({best['mcc']:.3f})")
    for entry in mcc_data:
        mcc = entry["mcc"]
        if mcc >= 0.8:
            quality = "🟢 Excelente"
        elif mcc >= 0.6:
            quality = "🟡 Bueno"
        elif mcc >= 0.4:
            quality = "🟠 Moderado"
        else:
            quality = "🔴 Bajo"
        st.write(f"- **{entry['model']}**: {mcc:.3f} — {quality}")


# ---------------------------------------------------------------------------
# Información técnica
# ---------------------------------------------------------------------------
def display_technical_info(model_info: dict, t: dict, technical_config: dict):
    with st.expander("⚙️ Información Técnica", expanded=False):
        st.write(f"- **Dataset:** {technical_config.get('DATASET_NAME', 'ISIC 2019')}")
        st.write(f"- **Tamaño del dataset:** {technical_config.get('DATASET_SIZE', '25,331')} imágenes")
        st.write(f"- **Accuracy del modelo:** {technical_config.get('MODEL_ACCURACY', 'N/A')}")
        st.write(f"- **Parámetros del modelo:** {model_info.get('total_params', 'N/A'):,}")
        st.write(f"- **Input shape:** {model_info.get('input_shape', 'N/A')}")


# ---------------------------------------------------------------------------
# Advertencia médica
# ---------------------------------------------------------------------------
def display_medical_disclaimer(t: dict):
    st.markdown("---")
    st.warning(
        f"### {t.get('disclaimer_title', '⚕️ Aviso Médico Importante')}\n\n"
        + t.get(
            "disclaimer_body",
            "Esta herramienta es únicamente para fines educativos y de investigación. "
            "**No reemplaza el diagnóstico médico profesional.** "
            "Consulte siempre a un dermatólogo certificado.",
        )
    )


# ---------------------------------------------------------------------------
# Generación de PDF
# ---------------------------------------------------------------------------
def display_pdf_generation_section(t: dict) -> bool:
    """Muestra el botón de generación de PDF y retorna True si fue presionado."""
    st.markdown("---")
    st.subheader("📄 " + t.get("generate_pdf", "Generar Reporte PDF"))
    return st.button(
        "📥 " + t.get("download_pdf", "Descargar Reporte PDF"),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Resultados McNemar
# ---------------------------------------------------------------------------
def display_mcnemar_results_table(mcnemar_results: list[dict], t: dict):
    if not mcnemar_results:
        st.info("Sin resultados de McNemar disponibles.")
        return
    import pandas as pd
    df = pd.DataFrame([
        {
            "Par de Modelos": f"{r['model_a']} vs {r['model_b']}",
            "χ²": f"{r['chi2']:.4f}",
            "p-valor": f"{r['p_value']:.6f}",
            "Significativo (α=0.05)": "✅ Sí" if r["significant"] else "❌ No",
        }
        for r in mcnemar_results
    ])
    st.dataframe(df, use_container_width=True)


def display_statistical_conclusions(mcnemar_results: list[dict], t: dict):
    if not mcnemar_results:
        return
    significant = [r for r in mcnemar_results if r["significant"]]
    total = len(mcnemar_results)
    st.markdown("**Conclusiones estadísticas:**")
    if significant:
        st.markdown(
            f"- {len(significant)}/{total} pares muestran diferencias **estadísticamente significativas** (p < 0.05)."
        )
    else:
        st.markdown(
            f"- Ningún par ({total}/{total}) muestra diferencias estadísticamente significativas (p ≥ 0.05)."
        )
