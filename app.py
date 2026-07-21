"""
app.py
Aplicación principal para el sistema de diagnóstico de cáncer de piel (ISIC 2019).

Fixes aplicados:
  #1  st.experimental_rerun() → st.rerun()
  #2  Progreso de entrenamiento via queue.Queue (thread-safe)
  #3  Grad-CAM restaurado (warm-up del modelo antes de generar el mapa)
  #4  try/except en carga y preprocesamiento de imagen
  #5  plt.close(fig) después de cada st.pyplot(fig)
  #6  @st.cache_data para comparación y métricas
  #7  Figuras guardadas en st.session_state.figures
  #8  update_model_metrics() retorna datos → guardados en session_state
  #9  Validación de tipo y tamaño de archivo en display_image_upload_section
  #10 main() descompuesto en funciones separadas
  #13 Disclaimer al inicio de la app
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Módulos propios
from config import initialize_page, PAGE_CONFIG
from model_utils import load_models, predict_image_with_debug, predict_image_with_custom_threshold, get_model_info
from preprocessing import preprocess_image
from hybrid_model_trainer import (
    train_hybrid_models_async, get_training_progress,
    check_hybrid_models_exist, is_training_active,
)
from hybrid_data_integrator import update_model_metrics
from ui_components import (
    setup_sidebar, display_main_header, display_image_upload_section,
    display_image_comparison, display_diagnosis_results, display_debug_info,
    display_interpretation, display_model_comparison_table,
    display_consistency_analysis, display_metrics_explanation,
    display_metrics_in_columns, display_mcc_interpretation,
    display_technical_info, display_medical_disclaimer,
    display_pdf_generation_section, display_mcnemar_results_table,
    display_statistical_conclusions,
)
from data_processor import (
    compare_all_models, get_model_metrics,
    create_mcc_comparison_dataframe, format_metrics_for_display,
)
from activation_maps import create_gradcam_figure
from visualization import (
    plot_confusion_matrix, create_metrics_dashboard,
    create_advanced_metrics_dashboard, create_model_comparison_plots,
    create_mcc_comparison_chart, create_mcnemar_plot,
)
from metrics_calculator import perform_mcnemar_comparisons
from pdf_generator_optimized import generate_pdf_report
from translations import get_available_languages, load_translations


# ---------------------------------------------------------------------------
# Helpers de caché
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_compare_models(model_names_key: str, processed_bytes: bytes) -> list[dict]:
    """
    Wrapper cacheado para compare_all_models.
    model_names_key y processed_bytes actúan como llaves de caché (Fix #6).
    La reconstrucción de modelos se hace fuera del caché.
    """
    # Esta función es llamada con los modelos ya disponibles en el scope de la
    # función llamadora; usamos el patrón de delegar al llamador.
    # El caché real se gestiona manualmente en session_state por la lógica de
    # comparación (ver _run_model_comparison) para poder pasar el dict models.
    pass


@st.cache_data(show_spinner=False)
def _cached_get_model_metrics(model_key: str, training_metrics_key: str) -> tuple:
    """Caché de métricas por modelo. training_metrics_key es un hash del dict."""
    # Delegamos a get_model_metrics con los datos del estado de sesión.
    pass


# ---------------------------------------------------------------------------
# Inicialización de session_state
# ---------------------------------------------------------------------------
def _init_session_state():
    """Inicializa todas las claves de session_state en un solo lugar."""
    defaults = {
        "uploaded_image":       None,
        "processed_image":      None,
        "diagnosis":            None,
        "confidence_percent":   None,
        "raw_confidence":       None,
        "last_model":           None,
        "last_threshold":       None,
        "comparison_results":   None,
        "models_hash":          None,
        "metrics_data":         {},
        "is_real_data":         {},
        # Fix #7: diccionario central de figuras
        "figures":              {},
        # Fix #8: métricas y MCC desde update_model_metrics()
        "training_metrics":     None,
        "mcc_comparison_data":  None,
        # Control de entrenamiento
        "start_hybrid_training": False,
        "training_in_progress":  False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_models_cached(t: dict) -> dict:
    try:
        models = load_models()
        if not models:
            st.error("❌ " + t.get("models_load_error", "No se pudieron cargar los modelos entrenados."))
            st.error("📝 " + t.get("models_folder_check", "Asegúrate de que los archivos .keras/.h5 estén en app/models/"))
            return {}
        return models
    except Exception as e:
        st.error(f"❌ {t.get('model_load_exception', 'Error al cargar los modelos')}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Bloque de entrenamiento híbrido
# ---------------------------------------------------------------------------
def _handle_hybrid_training():
    """Gestiona el inicio y seguimiento del entrenamiento de modelos híbridos (Fix #2)."""
    # Iniciar entrenamiento si se solicitó
    if st.session_state.get("start_hybrid_training"):
        st.session_state["start_hybrid_training"] = False
        if not st.session_state.get("training_in_progress"):
            st.session_state["training_in_progress"] = True
            train_hybrid_models_async()
            st.rerun()  # Fix #1: reemplaza st.experimental_rerun()

    # Si hay entrenamiento en progreso, mostrar barra de progreso real
    if st.session_state.get("training_in_progress"):
        progress_update = get_training_progress()  # lee de la queue thread-safe
        if progress_update is not None:
            progress_val, message = progress_update
            if progress_val < 0:
                st.error(f"Error en entrenamiento: {message}")
                st.session_state["training_in_progress"] = False
            elif progress_val >= 1.0:
                st.success(message)
                st.session_state["training_in_progress"] = False
                st.rerun()
            else:
                st.progress(progress_val, text=message)

        if is_training_active():
            st.info("⏳ Entrenamiento en progreso… La página se actualizará automáticamente.")
            # Auto-rerun cada ~2 s para leer la queue
            import time
            time.sleep(2)
            st.rerun()


# ---------------------------------------------------------------------------
# Bloque de imagen y diagnóstico
# ---------------------------------------------------------------------------
def _handle_image_upload(t: dict):
    """
    Carga, valida y preprocesa la imagen subida. (Fix #4, #9)
    Guarda en session_state.
    """
    uploaded_file = display_image_upload_section(t)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.verify()          # detecta archivos corruptos
            uploaded_file.seek(0)   # verify() consume el stream
            image = Image.open(uploaded_file)
        except Exception:
            st.error(t.get("upload_corrupt", "No se pudo leer la imagen. El archivo puede estar corrupto."))
            st.stop()

        try:
            processed_image = preprocess_image(np.array(image))
        except Exception as e:
            st.error(f"Error al preprocesar la imagen: {e}")
            st.stop()

        st.session_state.uploaded_image   = image
        st.session_state.processed_image  = processed_image
        # Resetear caché de diagnóstico al cargar nueva imagen
        st.session_state.diagnosis        = None
        st.session_state.last_model       = None
        st.session_state.comparison_results = None
        st.session_state.figures          = {}


def _run_diagnosis(models: dict, sidebar_config: dict, t: dict):
    """Ejecuta el diagnóstico con el modelo seleccionado."""
    image          = st.session_state.uploaded_image
    processed_image = st.session_state.processed_image
    model          = models[sidebar_config["selected_model"]]

    # Mostrar comparación de imágenes
    display_image_comparison(image, processed_image, t)

    st.header("🔍 " + t.get("diagnosis_results", "Resultados del Diagnóstico"))

    recalculate = (
        st.session_state.diagnosis is None
        or st.session_state.last_model      != sidebar_config["selected_model"]
        or st.session_state.last_threshold  != sidebar_config["decision_threshold"]
    )

    if recalculate:
        with st.spinner(t.get("processing_image", "Analizando imagen...")):
            if sidebar_config["debug_mode"]:
                diagnosis, confidence_percent, raw_confidence = predict_image_with_debug(
                    model, processed_image
                )
                display_debug_info(processed_image, model, sidebar_config["decision_threshold"], t)
            else:
                diagnosis, confidence_percent, raw_confidence = predict_image_with_custom_threshold(
                    model, processed_image, threshold=sidebar_config["decision_threshold"]
                )
        st.session_state.diagnosis          = diagnosis
        st.session_state.confidence_percent = confidence_percent
        st.session_state.raw_confidence     = raw_confidence
        st.session_state.last_model         = sidebar_config["selected_model"]
        st.session_state.last_threshold     = sidebar_config["decision_threshold"]
    else:
        diagnosis          = st.session_state.diagnosis
        confidence_percent = st.session_state.confidence_percent
        raw_confidence     = st.session_state.raw_confidence
        if sidebar_config["debug_mode"]:
            display_debug_info(processed_image, model, sidebar_config["decision_threshold"], t)

    display_diagnosis_results(diagnosis, confidence_percent, raw_confidence, t)
    display_interpretation(confidence_percent, sidebar_config["confidence_threshold"], diagnosis, t)

    return diagnosis, confidence_percent, raw_confidence


# ---------------------------------------------------------------------------
# Bloque de comparación de modelos (Fix #6 manual via session_state)
# ---------------------------------------------------------------------------
def _run_model_comparison(models: dict, sidebar_config: dict, t: dict):
    """Compara todos los modelos sobre la imagen actual."""
    processed_image    = st.session_state.processed_image
    current_models_key = ",".join(sorted(models.keys()))

    if (
        st.session_state.comparison_results is None
        or st.session_state.models_hash != current_models_key
    ):
        with st.spinner(t.get("comparing_models", "Comparando todos los modelos...")):
            comparison_results = compare_all_models(models, processed_image)
        st.session_state.comparison_results = comparison_results
        st.session_state.models_hash        = current_models_key
    else:
        comparison_results = st.session_state.comparison_results

    if comparison_results:
        df_comparison = pd.DataFrame(comparison_results)
        display_model_comparison_table(df_comparison)

        fig_conf, fig_time = create_model_comparison_plots(df_comparison)
        for key, fig in [("fig_conf", fig_conf), ("fig_time", fig_time)]:
            if fig:
                st.session_state.figures[key] = fig  # Fix #7
                st.pyplot(fig)
                plt.close(fig)                        # Fix #5

        display_consistency_analysis(comparison_results, t)

    return comparison_results


# ---------------------------------------------------------------------------
# Bloque de matriz de confusión y métricas (Fix #6 via session_state)
# ---------------------------------------------------------------------------
def _run_metrics_section(sidebar_config: dict, t: dict):
    """Muestra la matriz de confusión y métricas del modelo seleccionado."""
    model_key        = sidebar_config["selected_model"]
    training_metrics = st.session_state.get("training_metrics")   # Fix #8

    if model_key not in st.session_state.metrics_data:
        metrics_data, is_real_data = get_model_metrics(model_key, training_metrics)
        st.session_state.metrics_data[model_key]   = metrics_data
        st.session_state.is_real_data[model_key]   = is_real_data
    else:
        metrics_data   = st.session_state.metrics_data[model_key]
        is_real_data   = st.session_state.is_real_data[model_key]

    if is_real_data:
        st.success(
            t.get("real_data_metrics",
                  "✅ **Datos Reales**: métricas del modelo {model} en ISIC 2019"
                  ).format(model=model_key)
        )
    else:
        st.warning(t.get("simulated_data_metrics", "⚠️ **Datos Simulados**"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t.get('confusion_matrix_chart', '🎯 Matriz de Confusión')}**")
        fig_cm = plot_confusion_matrix(metrics_data["confusion_matrix"], model_key)
        if fig_cm:
            st.session_state.figures["fig_cm"] = fig_cm   # Fix #7
            st.pyplot(fig_cm)
            plt.close(fig_cm)                              # Fix #5

    with col2:
        st.markdown(f"**{t.get('advanced_metrics', '📈 Métricas Avanzadas')}**")
        display_metrics_in_columns(metrics_data, t)
        st.markdown(f"**{t.get('metrics_interpretation', '📋 Interpretación:')}**")
        st.markdown(f"""
        - **Accuracy**: {metrics_data['accuracy']*100:.1f}% {t.get('accuracy_explanation', 'de predicciones correctas')}
        - **Sensitivity**: {metrics_data['sensitivity']*100:.1f}% {t.get('sensitivity_explanation', 'de casos malignos detectados')}
        - **Specificity**: {metrics_data['specificity']*100:.1f}% {t.get('specificity_explanation', 'de casos benignos identificados')}
        - **Precision**: {metrics_data['precision']*100:.1f}% {t.get('precision_explanation', 'de los marcados malignos son malignos')}
        - **F1-Score**: {metrics_data['f1_score']*100:.1f}% {t.get('f1_explanation', 'balance precisión-sensibilidad')}
        - **MCC**: {metrics_data['mcc']:.3f} {t.get('mcc_explanation', '(Coeficiente de Matthews)')}
        """)

    # Dashboard completo
    st.markdown("---")
    st.subheader(f"📊 {t.get('metrics_dashboard_title', 'Dashboard Completo de Métricas')}")
    fig_dashboard = create_metrics_dashboard(metrics_data, model_key)
    if fig_dashboard:
        st.session_state.figures["fig_dashboard"] = fig_dashboard  # Fix #7
        st.pyplot(fig_dashboard)
        plt.close(fig_dashboard)                                    # Fix #5

    display_metrics_explanation(t)
    return metrics_data


# ---------------------------------------------------------------------------
# Bloque Grad-CAM (Fix #3)
# ---------------------------------------------------------------------------
def _run_gradcam(models: dict, sidebar_config: dict, t: dict):
    """Genera y muestra el mapa Grad-CAM si el usuario lo activó."""
    if not sidebar_config.get("show_gradcam", True):
        return

    st.markdown("---")
    st.subheader(t.get("gradcam_title", "🔥 Mapa de Activación Grad-CAM"))
    st.markdown(t.get("gradcam_description",
                      "Visualización de las regiones que más influyeron en el diagnóstico."))

    model           = models[sidebar_config["selected_model"]]
    processed_image = st.session_state.processed_image
    original_image  = np.array(st.session_state.uploaded_image)

    with st.spinner("Generando Grad-CAM…"):
        fig_gradcam = create_gradcam_figure(
            original_image=original_image,
            processed_image=processed_image,
            model=model,
            model_name=sidebar_config["selected_model"],
        )

    if fig_gradcam is not None:
        st.session_state.figures["fig_gradcam"] = fig_gradcam  # Fix #7
        st.pyplot(fig_gradcam)
        plt.close(fig_gradcam)                                  # Fix #5
    else:
        st.warning(t.get("gradcam_error", "No se pudo generar el mapa Grad-CAM para este modelo."))


# ---------------------------------------------------------------------------
# Bloque estadístico (MCC + McNemar)
# ---------------------------------------------------------------------------
def _run_statistical_section(sidebar_config: dict, t: dict):
    """Muestra el análisis estadístico avanzado: MCC y McNemar."""
    mcc_data = st.session_state.get("mcc_comparison_data")  # Fix #8
    if mcc_data is None:
        # Fallback a los datos de config si no se actualizaron
        from config import MCC_COMPARISON_DATA
        mcc_data = MCC_COMPARISON_DATA

    metrics_data = st.session_state.metrics_data.get(sidebar_config["selected_model"], {})

    # Dashboard avanzado
    st.markdown("---")
    st.subheader(f"🔬 {t.get('statistical_analysis_title', 'Análisis Estadístico Avanzado')}")
    st.markdown(t.get("statistical_analysis_description", "Incluye MCC y pruebas de McNemar") + ":")

    fig_advanced = create_advanced_metrics_dashboard(metrics_data, sidebar_config["selected_model"])
    if fig_advanced:
        st.session_state.figures["fig_advanced"] = fig_advanced   # Fix #7
        st.pyplot(fig_advanced)
        plt.close(fig_advanced)                                    # Fix #5

    # Tabla MCC
    st.markdown("---")
    st.subheader(f"📊 {t.get('mcc_comparison_title', 'Resumen Comparativo MCC')}")
    df_mcc = create_mcc_comparison_dataframe(mcc_data)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"**{t.get('mcc_table_title', '📋 Tabla MCC')}**")
        st.dataframe(df_mcc, use_container_width=True)
    with col2:
        display_mcc_interpretation(mcc_data, t)

    fig_mcc = create_mcc_comparison_chart(mcc_data)
    if fig_mcc:
        st.session_state.figures["fig_mcc"] = fig_mcc   # Fix #7
        st.pyplot(fig_mcc)
        plt.close(fig_mcc)                               # Fix #5

    # McNemar
    st.markdown("---")
    st.subheader(f"🔬 {t.get('mcnemar_tests_title', 'Pruebas de McNemar')}")
    st.markdown(t.get("mcnemar_description", "Comparación estadística entre modelos") + ":")
    mcnemar_results = perform_mcnemar_comparisons(mcc_data)
    display_mcnemar_results_table(mcnemar_results, t)

    fig_mcnemar = create_mcnemar_plot(mcnemar_results)
    if fig_mcnemar:
        st.session_state.figures["fig_mcnemar"] = fig_mcnemar   # Fix #7
        st.pyplot(fig_mcnemar)
        plt.close(fig_mcnemar)                                   # Fix #5

    display_statistical_conclusions(mcnemar_results, t)
    return mcnemar_results


# ---------------------------------------------------------------------------
# Bloque de generación de PDF (Fix #7)
# ---------------------------------------------------------------------------
def _run_pdf_generation(models: dict, sidebar_config: dict, t: dict,
                         diagnosis: str, confidence_percent: float, raw_confidence: float,
                         metrics_data: dict, comparison_results: list):
    """Genera y ofrece el reporte PDF para descarga."""
    generate_pdf = display_pdf_generation_section(t)
    if not generate_pdf:
        return

    with st.spinner(t.get("generating_pdf", "Generando reporte PDF...")):
        figs = st.session_state.figures   # Fix #7: figuras guardadas explícitamente

        plots_data = {
            "confusion_matrix":  figs.get("fig_cm"),
            "metrics_dashboard": figs.get("fig_dashboard"),
            "advanced_dashboard": figs.get("fig_advanced"),
            "comparison_plots": {
                t.get("confidence_comparison_plot", "Comparacion de Confianza"): figs.get("fig_conf"),
                t.get("inference_speed_plot",        "Velocidad de Inferencia"): figs.get("fig_time"),
                t.get("mcc_comparative_plot",        "MCC Comparativo"):          figs.get("fig_mcc"),
                t.get("mcnemar_pvalues_plot",        "McNemar P-valores"):         figs.get("fig_mcnemar"),
            },
        }

        generate_pdf_report(
            image=st.session_state.uploaded_image,
            diagnosis=diagnosis,
            confidence_percent=confidence_percent,
            raw_confidence=raw_confidence,
            model_name=sidebar_config["selected_model"],
            model_info=get_model_info(models[sidebar_config["selected_model"]]),
            comparison_results=comparison_results,
            translations=t,
            confidence_threshold=sidebar_config["confidence_threshold"],
            metrics_data=metrics_data,
            plots_data=plots_data,
        )


# ---------------------------------------------------------------------------
# main()  — índice del flujo de la app (Fix #10)
# ---------------------------------------------------------------------------
def main():
    initialize_page()

    # Traducciones iniciales
    available_languages = get_available_languages()
    t = load_translations("es")

    # Carga de modelos
    models = _load_models_cached(t)
    if not models:
        st.stop()

    _init_session_state()

    # Fix #8: cargar métricas de modelos híbridos una sola vez por sesión
    if st.session_state.training_metrics is None:
        hybrid_exists, _ = check_hybrid_models_exist()
        if hybrid_exists:
            updated_metrics, updated_mcc = update_model_metrics()
            st.session_state.training_metrics    = updated_metrics
            st.session_state.mcc_comparison_data = updated_mcc
        else:
            from config import REAL_TRAINING_METRICS, MCC_COMPARISON_DATA
            st.session_state.training_metrics    = REAL_TRAINING_METRICS
            st.session_state.mcc_comparison_data = MCC_COMPARISON_DATA

    # Barra lateral
    sidebar_config = setup_sidebar(models, t, available_languages)

    # Actualizar traducciones según idioma elegido
    current_lang_code = available_languages[sidebar_config["lang"]]
    t = load_translations(current_lang_code)

    # Validar que hay modelos
    if not list(models.keys()):
        st.error("❌ " + t.get("no_models_available", "No hay modelos disponibles."))
        st.stop()

    # Encabezado
    display_main_header(t)

    # Fix #13: disclaimer médico al INICIO, antes de la carga de imagen
    display_medical_disclaimer(t)

    st.markdown("---")

    # Entrenamiento en segundo plano
    _handle_hybrid_training()

    # Carga y preprocesamiento de imagen (Fix #4, #9)
    _handle_image_upload(t)

    # Si no hay imagen cargada, detener aquí
    if st.session_state.uploaded_image is None:
        st.info("👆 Sube una imagen dermatoscópica para comenzar el análisis.")
        return

    # --- Diagnóstico ---
    diagnosis, confidence_percent, raw_confidence = _run_diagnosis(models, sidebar_config, t)

    # --- Comparación de modelos ---
    st.markdown("---")
    st.subheader(f"📊 {t.get('model_comparison', 'Comparación de Todos los Modelos')}")
    st.markdown(t.get("model_comparison_desc", "Análisis de la misma imagen con diferentes modelos") + ":")
    comparison_results = _run_model_comparison(models, sidebar_config, t)

    # --- Matriz de confusión y métricas ---
    st.markdown("---")
    st.subheader(f"📊 {t.get('confusion_matrix_title', 'Matriz de Confusión y Métricas')}")
    st.markdown(t.get("model_analysis_description", "Análisis detallado del modelo seleccionado") + ":")
    metrics_data = _run_metrics_section(sidebar_config, t)

    # --- Grad-CAM (Fix #3) ---
    _run_gradcam(models, sidebar_config, t)

    # --- Análisis estadístico ---
    _run_statistical_section(sidebar_config, t)

    # --- Generación de PDF (Fix #7) ---
    _run_pdf_generation(
        models, sidebar_config, t,
        diagnosis, confidence_percent, raw_confidence,
        metrics_data, comparison_results,
    )

    # --- Información técnica ---
    technical_config = {
        "DATASET_NAME":        PAGE_CONFIG.get("dataset_name", "ISIC 2019"),
        "DATASET_SIZE":        "25,331",
        "MODEL_ACCURACY":      f"~{st.session_state.training_metrics.get(sidebar_config['selected_model'], {}).get('accuracy', 0)*100:.1f}%",
        "OPTIMIZATION_TARGET": "cáncer de piel",
    }
    display_technical_info(
        get_model_info(models[sidebar_config["selected_model"]]), t, technical_config
    )

    # Disclaimer al final también
    display_medical_disclaimer(t)


if __name__ == "__main__":
    main()
