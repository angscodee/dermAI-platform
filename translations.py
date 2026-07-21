"""
translations.py
Soporte multiidioma para la aplicación.
"""

_TRANSLATIONS = {
    "es": {
        # General
        "models_load_error": "No se pudieron cargar los modelos entrenados.",
        "models_folder_check": "Asegúrate de que los archivos .h5 o .keras estén en la carpeta app/models/",
        "model_load_exception": "Error al cargar los modelos",
        "no_models_available": "No hay modelos disponibles. Verifica que los modelos entrenados estén en app/models/",
        "processing_image": "Analizando imagen...",
        "comparing_models": "Comparando todos los modelos...",
        "generating_pdf": "Generando reporte PDF...",
        # Upload
        "upload_title": "Cargar Imagen Dermatoscópica",
        "upload_instructions": "Sube una imagen de la lesión cutánea para su análisis",
        "upload_label": "Selecciona una imagen (.jpg, .jpeg o .png)",
        "upload_invalid_type": "Tipo de archivo no permitido. Usa .jpg, .jpeg o .png.",
        "upload_too_large": "El archivo supera el límite de 10 MB. Por favor, reduce el tamaño.",
        "upload_corrupt": "No se pudo leer la imagen. El archivo puede estar corrupto.",
        # Diagnosis
        "diagnosis_results": "Resultados del Diagnóstico",
        "diagnosis_malignant": "⚠️ POSIBLE LESIÓN MALIGNA",
        "diagnosis_benign": "✅ LESIÓN PROBABLEMENTE BENIGNA",
        # Comparison
        "model_comparison": "Comparación de Todos los Modelos",
        "model_comparison_desc": "Resultados de análisis de la misma imagen con diferentes modelos",
        # Confusion matrix
        "confusion_matrix_title": "Matriz de Confusión y Métricas",
        "model_analysis_description": "Análisis detallado del rendimiento del modelo seleccionado",
        "real_data_metrics": "✅ **Datos Reales de Entrenamiento**: Mostrando métricas reales del modelo {model} en el dataset ISIC 2019",
        "simulated_data_metrics": "⚠️ **Datos Simulados**: Usando datos de ejemplo para demostración",
        "confusion_matrix_chart": "🎯 Matriz de Confusión",
        "advanced_metrics": "📈 Métricas de Rendimiento Avanzadas",
        "metrics_interpretation": "📋 Interpretación:",
        "accuracy_explanation": "de las predicciones son correctas",
        "sensitivity_explanation": "de los casos malignos son detectados",
        "specificity_explanation": "de los casos benignos son correctamente identificados",
        "precision_explanation": "de los casos clasificados como malignos son realmente malignos",
        "f1_explanation": "es el balance entre precisión y sensibilidad",
        "mcc_explanation": "(Coeficiente de Matthews - balanceado para clases desequilibradas)",
        # Dashboard
        "metrics_dashboard_title": "Dashboard Completo de Métricas",
        # Grad-CAM
        "gradcam_title": "🔥 Mapa de Activación Grad-CAM",
        "gradcam_description": "Visualización de las regiones de la imagen que más influyeron en el diagnóstico",
        "gradcam_error": "No se pudo generar el mapa Grad-CAM para este modelo.",
        # Statistical
        "statistical_analysis_title": "Análisis Estadístico Avanzado",
        "statistical_analysis_description": "Incluyendo Coeficiente de Matthews y Pruebas de McNemar",
        "mcc_comparison_title": "Resumen Comparativo de Coeficientes de Matthews (MCC)",
        "mcc_comparison_description": "Comparación de todos los modelos basada en el Coeficiente de Matthews",
        "mcc_table_title": "📋 Tabla de Resumen - Coeficientes de Matthews",
        "mcnemar_tests_title": "Pruebas Estadísticas de McNemar",
        "mcnemar_description": "Comparación estadística entre modelos",
        # PDF
        "confidence_comparison_plot": "Comparacion de Confianza",
        "inference_speed_plot": "Velocidad de Inferencia",
        "mcc_comparative_plot": "MCC Comparativo",
        "mcnemar_pvalues_plot": "McNemar P-valores",
        # Disclaimer
        "disclaimer_title": "⚕️ Aviso Médico Importante",
        "disclaimer_body": (
            "Esta herramienta es únicamente para fines educativos y de investigación. "
            "**No reemplaza el diagnóstico médico profesional.** "
            "Consulte siempre a un dermatólogo certificado para el diagnóstico y tratamiento "
            "de lesiones cutáneas. Los resultados de esta IA deben considerarse como una "
            "herramienta de apoyo, no como un diagnóstico definitivo."
        ),
    },
    "en": {
        "models_load_error": "Could not load trained models.",
        "models_folder_check": "Make sure .h5 or .keras files are in the app/models/ folder",
        "model_load_exception": "Error loading models",
        "no_models_available": "No models available. Check that trained models are in app/models/",
        "processing_image": "Analyzing image...",
        "comparing_models": "Comparing all models...",
        "generating_pdf": "Generating PDF report...",
        "upload_title": "Upload Dermoscopic Image",
        "upload_instructions": "Upload an image of the skin lesion for analysis",
        "upload_label": "Select an image (.jpg, .jpeg or .png)",
        "upload_invalid_type": "File type not allowed. Use .jpg, .jpeg or .png.",
        "upload_too_large": "File exceeds the 10 MB limit. Please reduce the size.",
        "upload_corrupt": "Could not read the image. The file may be corrupted.",
        "diagnosis_results": "Diagnosis Results",
        "diagnosis_malignant": "⚠️ POSSIBLE MALIGNANT LESION",
        "diagnosis_benign": "✅ PROBABLY BENIGN LESION",
        "model_comparison": "Comparison of All Models",
        "model_comparison_desc": "Analysis results of the same image with different models",
        "confusion_matrix_title": "Confusion Matrix and Metrics",
        "model_analysis_description": "Detailed performance analysis of the selected model",
        "real_data_metrics": "✅ **Real Training Data**: Showing real metrics for model {model} on ISIC 2019 dataset",
        "simulated_data_metrics": "⚠️ **Simulated Data**: Using sample data for demonstration",
        "confusion_matrix_chart": "🎯 Confusion Matrix",
        "advanced_metrics": "📈 Advanced Performance Metrics",
        "metrics_interpretation": "📋 Interpretation:",
        "accuracy_explanation": "of predictions are correct",
        "sensitivity_explanation": "of malignant cases are detected",
        "specificity_explanation": "of benign cases are correctly identified",
        "precision_explanation": "of cases classified as malignant are truly malignant",
        "f1_explanation": "is the balance between precision and recall",
        "mcc_explanation": "(Matthews Correlation Coefficient - balanced for imbalanced classes)",
        "metrics_dashboard_title": "Complete Metrics Dashboard",
        "gradcam_title": "🔥 Grad-CAM Activation Map",
        "gradcam_description": "Visualization of image regions that most influenced the diagnosis",
        "gradcam_error": "Could not generate Grad-CAM map for this model.",
        "statistical_analysis_title": "Advanced Statistical Analysis",
        "statistical_analysis_description": "Including Matthews Coefficient and McNemar Tests",
        "mcc_comparison_title": "Comparative Summary of Matthews Coefficients (MCC)",
        "mcc_comparison_description": "Comparison of all models based on the Matthews Coefficient",
        "mcc_table_title": "📋 Summary Table - Matthews Coefficients",
        "mcnemar_tests_title": "McNemar Statistical Tests",
        "mcnemar_description": "Statistical comparison between models",
        "confidence_comparison_plot": "Confidence Comparison",
        "inference_speed_plot": "Inference Speed",
        "mcc_comparative_plot": "MCC Comparative",
        "mcnemar_pvalues_plot": "McNemar P-values",
        "disclaimer_title": "⚕️ Important Medical Notice",
        "disclaimer_body": (
            "This tool is for educational and research purposes only. "
            "**It does not replace professional medical diagnosis.** "
            "Always consult a certified dermatologist for the diagnosis and treatment "
            "of skin lesions. The results from this AI should be considered as a "
            "support tool, not a definitive diagnosis."
        ),
    },
}


def get_available_languages() -> dict:
    """Retorna un dict {nombre_display: código_iso}."""
    return {
        "Español": "es",
        "English": "en",
    }


def load_translations(lang_code: str) -> dict:
    """
    Carga las traducciones para el código de idioma dado.
    Hace fallback a español si el código no existe.
    """
    return _TRANSLATIONS.get(lang_code, _TRANSLATIONS["es"])
