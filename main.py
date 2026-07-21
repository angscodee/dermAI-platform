CODIGO PROGRAMA:
# app.py
"""
Aplicación principal para el sistema de diagnóstico de cáncer de piel
"""

import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Importar módulos propios
from config import initialize_page, MCC_COMPARISON_DATA, PAGE_CONFIG, REAL_TRAINING_METRICS
from model_utils import load_models, predict_image, predict_image_with_debug, predict_image_with_custom_threshold, get_model_info
from preprocessing import preprocess_image
from hybrid_model_trainer import train_hybrid_models_async, get_training_progress, check_hybrid_models_exist
from hybrid_data_integrator import update_model_metrics
from ui_components import (
    setup_sidebar, display_main_header, display_image_upload_section, display_image_comparison,
    display_diagnosis_results, display_debug_info, display_interpretation, display_model_comparison_table,
    display_consistency_analysis, display_metrics_explanation, display_metrics_in_columns,
    display_mcc_interpretation, display_technical_info, display_medical_disclaimer,
    display_pdf_generation_section, display_mcnemar_results_table, display_statistical_conclusions
)
from data_processor import (
    compare_all_models, analyze_consistency, get_model_metrics,
    create_mcc_comparison_dataframe, format_metrics_for_display
)
# La importación de activation_maps ha sido eliminada debido a errores con generate_gradcam
from visualization import (
    plot_confusion_matrix, create_metrics_dashboard, create_advanced_metrics_dashboard,
    create_model_comparison_plots, create_mcc_comparison_chart, create_mcnemar_plot
)
from metrics_calculator import perform_mcnemar_comparisons
from pdf_generator_optimized import generate_pdf_report
from translations import get_available_languages, load_translations

def main():
    """Función principal de la aplicación"""
    # Inicializar la configuración de la página
    initialize_page()
    
    # Configuración inicial para traducciones (español por defecto)
    available_languages = get_available_languages()
    t = load_translations('es')
    
    # Cargar modelos entrenados
    @st.cache_resource
    def load_models_cached():
        try:
            models = load_models()
            if not models:
                st.error("❌ " + t.get('models_load_error', "No se pudieron cargar los modelos entrenados."))
                st.error("📝 " + t.get('models_folder_check', "Asegúrate de que los archivos .h5 o .keras estén en la carpeta app/models/"))
                return {}
            return models
        except Exception as e:
            st.error(f"❌ {t.get('model_load_exception', 'Error al cargar los modelos')}: {str(e)}")
            return {}

    # Verificar y actualizar métricas de modelos híbridos si existen
    hybrid_models_exist, hybrid_model_names = check_hybrid_models_exist()
    if hybrid_models_exist:
        # Actualizar métricas y datos MCC si existen modelos híbridos
        global REAL_TRAINING_METRICS, MCC_COMPARISON_DATA
        updated_metrics, updated_mcc = update_model_metrics()
        
        # Solo actualizar si se encontraron métricas
        if len(updated_metrics) > len(REAL_TRAINING_METRICS):
            REAL_TRAINING_METRICS = updated_metrics
        if len(updated_mcc) > len(MCC_COMPARISON_DATA):
            MCC_COMPARISON_DATA = updated_mcc

    # Cargar modelos
    models = load_models_cached()
    model_names = list(models.keys())

    # Manejar el entrenamiento de modelos híbridos
    if 'start_hybrid_training' in st.session_state and st.session_state['start_hybrid_training']:
        # Resetear la bandera
        st.session_state['start_hybrid_training'] = False
        
        # Iniciar entrenamiento si no está en progreso
        if 'training_in_progress' not in st.session_state or not st.session_state['training_in_progress']:
            st.session_state['training_in_progress'] = True
            
            # Función de callback para actualizar el progreso
            def update_progress(message, progress, error=False):
                # Esta función se ejecutará desde otro hilo, los cambios
                # en st.session_state serán visibles en la próxima rerenderización
                pass
            
            # Iniciar entrenamiento en segundo plano
            train_hybrid_models_async(progress_callback=update_progress)
            st.experimental_rerun()

    if not model_names:
        st.error("❌ " + t.get('no_models_available', "No hay modelos disponibles. Verifica que los modelos entrenados estén en app/models/"))
        st.stop()
    
    # Configurar la barra lateral y obtener la configuración del usuario
    sidebar_config = setup_sidebar(models, t, available_languages)
    
    # Actualizar las traducciones según el idioma seleccionado
    current_lang_code = available_languages[sidebar_config['lang']]
    t = load_translations(current_lang_code)
    
    # Mostrar encabezado principal
    display_main_header(t)
    
    # Sección de carga de imagen
    uploaded_file = display_image_upload_section(t)
    
    # Inicializar variables de sesión si no existen
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
        st.session_state.processed_image = None
    
    # Procesar nueva imagen si se ha cargado
    if uploaded_file is not None:
        # Guardar la imagen original en la sesión
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        
        # Preprocesamiento y guardado en sesión
        processed_image = preprocess_image(np.array(image))
        st.session_state.processed_image = processed_image
    
    # Procesar la imagen guardada (ya sea nueva o de una carga anterior)
    if st.session_state.uploaded_image is not None:
        # Obtener imagen y su versión procesada de la sesión
        image = st.session_state.uploaded_image
        processed_image = st.session_state.processed_image
        
        # Mostrar comparación de imágenes
        display_image_comparison(image, processed_image, t)
        
        # Realizar predicción con el modelo seleccionado
        st.header("🔍 " + t.get('diagnosis_results', "Resultados del Diagnóstico"))
        
        with st.spinner(t.get('processing_image', "Analizando imagen...")):
            model = models[sidebar_config['selected_model']]
            
            # Verificamos si necesitamos recalcular el diagnóstico o usar el guardado
            recalculate = False
            
            # Inicializar variables de sesión para diagnóstico si no existen
            if 'diagnosis' not in st.session_state:
                st.session_state.diagnosis = None
                st.session_state.confidence_percent = None
                st.session_state.raw_confidence = None
                st.session_state.last_model = None
                st.session_state.last_threshold = None
                recalculate = True
            
            # Recalcular si cambió el modelo o el umbral de decisión
            if (st.session_state.last_model != sidebar_config['selected_model'] or 
                st.session_state.last_threshold != sidebar_config['decision_threshold']):
                recalculate = True
            
            if recalculate:
                if sidebar_config['debug_mode']:
                    # Usar función de debug
                    diagnosis, confidence_percent, raw_confidence = predict_image_with_debug(model, processed_image)
                    
                    # Mostrar información de debug
                    display_debug_info(processed_image, model, sidebar_config['decision_threshold'], t)
                else:
                    # Usar función con umbral personalizado
                    diagnosis, confidence_percent, raw_confidence = predict_image_with_custom_threshold(
                        model, processed_image, threshold=sidebar_config['decision_threshold']
                    )
                
                # Guardamos los resultados en la sesión
                st.session_state.diagnosis = diagnosis
                st.session_state.confidence_percent = confidence_percent
                st.session_state.raw_confidence = raw_confidence
                st.session_state.last_model = sidebar_config['selected_model']
                st.session_state.last_threshold = sidebar_config['decision_threshold']
            else:
                # Usamos los resultados guardados
                diagnosis = st.session_state.diagnosis
                confidence_percent = st.session_state.confidence_percent
                raw_confidence = st.session_state.raw_confidence
                
                # Si estamos en modo debug, mostramos la información de debug
                if sidebar_config['debug_mode']:
                    display_debug_info(processed_image, model, sidebar_config['decision_threshold'], t)
        
        # Mostrar resultados con mejor diseño
        display_diagnosis_results(diagnosis, confidence_percent, raw_confidence, t)
        
        # Interpretación de resultados
        display_interpretation(confidence_percent, sidebar_config['confidence_threshold'], diagnosis, t)
        
        # COMPARACIÓN DE TODOS LOS MODELOS
        st.markdown("---")
        st.subheader("📊 " + t.get('model_comparison', "Comparación de Todos los Modelos"))
        st.markdown(t.get('model_comparison_desc', "Resultados de análisis de la misma imagen con diferentes modelos:"))
        
        # Inicializar variable de sesión para comparación si no existe
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = None
            st.session_state.models_hash = None
        
        # Generamos un hash de los modelos para detectar cambios
        current_models_hash = ','.join(sorted(models.keys()))
        
        # Verificar si necesitamos recalcular la comparación
        recalculate_comparison = (st.session_state.comparison_results is None or 
                                 st.session_state.models_hash != current_models_hash)
        
        # Realizar predicciones con todos los modelos si es necesario
        if recalculate_comparison:
            with st.spinner(t.get('comparing_models', "Comparando todos los modelos...")):
                comparison_results = compare_all_models(models, processed_image)
                # Guardar resultados en la sesión
                st.session_state.comparison_results = comparison_results
                st.session_state.models_hash = current_models_hash
        else:
            # Usar resultados guardados
            comparison_results = st.session_state.comparison_results
        
        # Mostrar tabla de comparación
        if comparison_results:
            df_comparison = pd.DataFrame(comparison_results)
            display_model_comparison_table(df_comparison)
            
            # Gráficos de comparación
            fig_confianza, fig_tiempo = create_model_comparison_plots(df_comparison)
            if fig_confianza:
                st.pyplot(fig_confianza)
            if fig_tiempo:
                st.pyplot(fig_tiempo)
            
            # Análisis de consistencia
            display_consistency_analysis(comparison_results, t)
        
        # MATRIZ DE CONFUSIÓN Y MÉTRICAS
        st.markdown("---")
        confusion_matrix_title = t.get('confusion_matrix_title', "Matriz de Confusión y Métricas")
        st.subheader("📊 " + confusion_matrix_title)
        
        analysis_desc = t.get('model_analysis_description', "Análisis detallado del rendimiento del modelo seleccionado")
        st.markdown(f"{analysis_desc}:")
        
        # Inicializar variables de sesión para métricas si no existen
        if 'metrics_data' not in st.session_state:
            st.session_state.metrics_data = {}
            st.session_state.is_real_data = {}
        
        # Verificar si necesitamos calcular métricas para este modelo
        model_key = sidebar_config['selected_model']
        if model_key not in st.session_state.metrics_data:
            # Obtener métricas del modelo seleccionado
            metrics_data, is_real_data = get_model_metrics(model_key)
            # Guardar en sesión
            st.session_state.metrics_data[model_key] = metrics_data
            st.session_state.is_real_data[model_key] = is_real_data
        else:
            # Usar métricas guardadas
            metrics_data = st.session_state.metrics_data[model_key]
            is_real_data = st.session_state.is_real_data[model_key]
        
        if is_real_data:
            real_data_msg = t.get('real_data_metrics', "✅ **Datos Reales de Entrenamiento**: Mostrando métricas reales del modelo {model} en el dataset ISIC 2019")
            st.success(real_data_msg.format(model=sidebar_config['selected_model']))
        else:
            simulated_data_msg = t.get('simulated_data_metrics', "⚠️ **Datos Simulados**: Usando datos de ejemplo para demostración")
            st.warning(simulated_data_msg)
        
        # Mostrar matriz de confusión
        col1, col2 = st.columns(2)
        
        with col1:
            confusion_matrix_label = t.get('confusion_matrix_chart', "🎯 Matriz de Confusión")
            st.markdown(f"**{confusion_matrix_label}**")
            fig_cm = plot_confusion_matrix(metrics_data['confusion_matrix'], sidebar_config['selected_model'])
            if fig_cm:
                st.pyplot(fig_cm)
        
        with col2:
            advanced_metrics_label = t.get('advanced_metrics', "📈 Métricas de Rendimiento Avanzadas")
            st.markdown(f"**{advanced_metrics_label}**")
            display_metrics_in_columns(metrics_data, t)
            
            # Interpretación de métricas
            interpretation_label = t.get('metrics_interpretation', "📋 Interpretación:")
            st.markdown(f"**{interpretation_label}**")
            formatted_metrics = format_metrics_for_display(metrics_data)
            
            # Métricas formateadas con valores
            st.markdown(f"""
            - **Accuracy**: {metrics_data['accuracy']*100:.1f}% {t.get('accuracy_explanation', "de las predicciones son correctas")}
            - **Sensitivity**: {metrics_data['sensitivity']*100:.1f}% {t.get('sensitivity_explanation', "de los casos malignos son detectados")}
            - **Specificity**: {metrics_data['specificity']*100:.1f}% {t.get('specificity_explanation', "de los casos benignos son correctamente identificados")}
            - **Precision**: {metrics_data['precision']*100:.1f}% {t.get('precision_explanation', "de los casos clasificados como malignos son realmente malignos")}
            - **F1-Score**: {metrics_data['f1_score']*100:.1f}% {t.get('f1_explanation', "es el balance entre precisión y sensibilidad")}
            - **MCC**: {metrics_data['mcc']:.3f} {t.get('mcc_explanation', "(Coeficiente de Matthews - balanceado para clases desequilibradas)")}
            """)
        
        # Dashboard completo de métricas
        st.markdown("---")
        dashboard_title = t.get('metrics_dashboard_title', "Dashboard Completo de Métricas")
        st.subheader(f"📊 {dashboard_title}")
        
        fig_dashboard = create_metrics_dashboard(metrics_data, sidebar_config['selected_model'])
        if fig_dashboard:
            st.pyplot(fig_dashboard)
        
        # La sección de visualización de mapas de activación ha sido eliminada debido al error:
        # "Error al generar el mapa de activación Grad-CAM: The layer sequential has never been called and thus has no defined output."
        
        # Explicación de la matriz de confusión
        display_metrics_explanation(t)
        
        # Análisis Estadístico Avanzado
        st.markdown("---")
        statistical_title = t.get('statistical_analysis_title', "Análisis Estadístico Avanzado")
        st.subheader(f"🔬 {statistical_title}")
        
        statistical_desc = t.get('statistical_analysis_description', "Incluyendo Coeficiente de Matthews y Pruebas de McNemar")
        st.markdown(f"{statistical_desc}:")
        
        # Crear dashboard avanzado con MCC
        fig_advanced = create_advanced_metrics_dashboard(metrics_data, sidebar_config['selected_model'])
        if fig_advanced:
            st.pyplot(fig_advanced)
        
        # Tabla de Resumen MCC y Gráfico Comparativo
        st.markdown("---")
        mcc_title = t.get('mcc_comparison_title', "Resumen Comparativo de Coeficientes de Matthews (MCC)")
        st.subheader(f"📊 {mcc_title}")
        
        mcc_desc = t.get('mcc_comparison_description', "Comparación de todos los modelos basada en el Coeficiente de Matthews")
        st.markdown(f"{mcc_desc}:")
        
        # Crear DataFrame para la tabla
        df_mcc = create_mcc_comparison_dataframe(MCC_COMPARISON_DATA)
        
        # Mostrar tabla con formato mejorado
        col1, col2 = st.columns([3, 2])
        
        with col1:
            mcc_table_title = t.get('mcc_table_title', "📋 Tabla de Resumen - Coeficientes de Matthews")
            st.markdown(f"**{mcc_table_title}**")
            st.dataframe(df_mcc, use_container_width=True)
        
        with col2:
            display_mcc_interpretation(MCC_COMPARISON_DATA, t)
        
        # Gráfico MCC
        fig_mcc = create_mcc_comparison_chart(MCC_COMPARISON_DATA)
        if fig_mcc:
            st.pyplot(fig_mcc)
        
        # Análisis Estadístico - Pruebas de McNemar
        st.markdown("---")
        mcnemar_title = t.get('mcnemar_tests_title', "Pruebas Estadísticas de McNemar")
        st.subheader(f"🔬 {mcnemar_title}")
        
        mcnemar_desc = t.get('mcnemar_description', "Comparación estadística entre modelos")
        st.markdown(f"{mcnemar_desc}:")
        
        # Generar resultados de McNemar usando los datos de MCC para los nombres de modelos
        mcnemar_results = perform_mcnemar_comparisons(MCC_COMPARISON_DATA)
        
        # Mostrar tabla de resultados
        display_mcnemar_results_table(mcnemar_results, t)
        
        # Gráfico de p-valores
        fig_mcnemar = create_mcnemar_plot(mcnemar_results)
        if fig_mcnemar:
            st.pyplot(fig_mcnemar)
        
        # Conclusiones estadísticas
        display_statistical_conclusions(mcnemar_results, t)
        
        # Generar reporte PDF
        generate_pdf = display_pdf_generation_section(t)
        
        if generate_pdf:
            generating_pdf_msg = t.get('generating_pdf', "Generando reporte PDF...")
            with st.spinner(generating_pdf_msg):
                # Preparar datos para el PDF
                confidence_comparison_label = t.get('confidence_comparison_plot', "Comparacion de Confianza")
                inference_speed_label = t.get('inference_speed_plot', "Velocidad de Inferencia")
                mcc_comparative_label = t.get('mcc_comparative_plot', "MCC Comparativo")
                mcnemar_pvalues_label = t.get('mcnemar_pvalues_plot', "McNemar P-valores")
                
                plots_data = {
                    'confusion_matrix': fig_cm if 'fig_cm' in locals() else None,
                    'metrics_dashboard': fig_dashboard if 'fig_dashboard' in locals() else None,
                    'advanced_dashboard': fig_advanced if 'fig_advanced' in locals() else None,
                    'comparison_plots': {
                        confidence_comparison_label: fig_confianza if 'fig_confianza' in locals() else None,
                        inference_speed_label: fig_tiempo if 'fig_tiempo' in locals() else None,
                        mcc_comparative_label: fig_mcc if 'fig_mcc' in locals() else None,
                        mcnemar_pvalues_label: fig_mcnemar if 'fig_mcnemar' in locals() else None
                    }
                }
                
                # Generar PDF
                generate_pdf_report(
                    image=image,
                    diagnosis=diagnosis,
                    confidence_percent=confidence_percent,
                    raw_confidence=raw_confidence,
                    model_name=sidebar_config['selected_model'],
                    model_info=get_model_info(models[sidebar_config['selected_model']]),
                    comparison_results=comparison_results,
                    translations=t,
                    confidence_threshold=sidebar_config['confidence_threshold'],
                    metrics_data=metrics_data,
                    plots_data=plots_data
                )
        
        # Información técnica
        # Crear configuración para información técnica
        technical_config = {
            "DATASET_NAME": PAGE_CONFIG.get("dataset_name", "ISIC 2019"),
            "DATASET_SIZE": "25,331",
            "MODEL_ACCURACY": f"~{REAL_TRAINING_METRICS[sidebar_config['selected_model']]['accuracy']*100:.1f}%",
            "OPTIMIZATION_TARGET": "cáncer de piel"
        }
        display_technical_info(get_model_info(models[sidebar_config['selected_model']]), t, technical_config)
        
        # Advertencia médica
        display_medical_disclaimer(t)

if __name__ == "__main__":
    main()



