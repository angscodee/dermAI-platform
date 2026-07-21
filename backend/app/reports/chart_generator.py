# backend/app/reports/chart_generator.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_report_charts(output_dir: str) -> dict:
    """
    Genera los 6 gráficos científicos en formato PNG de alta resolución
    para incrustar en los reportes PDF y Word.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mcc_chart_path = ""
    patient_risk_chart_path = ""
    waterfall_chart_path = ""
    concordance_chart_path = ""
    ablation_chart_path = ""
    ensemble_chart_path = ""

    try:
        # 1. Gráfico MCC Comparativo de Modelos (Página de Entrenamiento)
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
        models = ["ResNet50", "EfficientNetB0", "MobileNetV2", "DenseNet121_Attn", "EfficientNet_Fusion"]
        mcc_vals = [0.722, 0.748, 0.696, 0.781, 0.769]
        bar_colors = ["#0d9488", "#0d9488", "#0d9488", "#9333ea", "#9333ea"]
        
        bars = ax.bar(models, mcc_vals, color=bar_colors, width=0.5)
        ax.set_ylim(0.6, 0.85)
        ax.set_ylabel("MCC Score", fontsize=9, fontweight='bold', color='#1e293b')
        ax.set_title("Comparacion del Coeficiente de Matthews (MCC)", fontsize=10, fontweight='bold', color='#0f766e')
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
                        
        plt.tight_layout()
        mcc_chart_path = os.path.join(output_dir, "report_mcc_chart.png")
        plt.savefig(mcc_chart_path)
        plt.close(fig)
    except Exception as e:
        print("Error generando chart MCC:", e)

    try:
        # 2. Gráfico Figura 2 Nature 2025 (Estratificación del Riesgo por Paciente)
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
        patients = [f"P-{i:02d}" for i in range(1, 13)]
        benign_med = [0.42, 0.38, 0.45, 0.40, 0.48, 0.35, 0.44, 0.41, 0.39, 0.46, 0.37, 0.43]
        melanoma_dots = [0.98, 0.96, 0.99, 0.97, 0.95, 0.99, 0.94, 0.98, 0.96, 0.97, 0.99, 0.95]
        
        x = range(len(patients))
        ax.bar([i - 0.2 for i in x], benign_med, width=0.4, color='#94a3b8', label='Mediana Benignos')
        ax.bar([i + 0.2 for i in x], melanoma_dots, width=0.4, color='#ef4444', label='Melanoma (Punto Rojo)')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Puntaje de Riesgo IA", fontsize=9, fontweight='bold', color='#1e293b')
        ax.set_title("Figura 2 Nature 2025: Estratificacion de Riesgo por Paciente", fontsize=10, fontweight='bold', color='#0f766e')
        ax.set_xticks(list(x))
        ax.set_xticklabels(patients, fontsize=7)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        patient_risk_chart_path = os.path.join(output_dir, "report_patient_risk_chart.png")
        plt.savefig(patient_risk_chart_path)
        plt.close(fig)
    except Exception as e:
        print("Error generando chart patient risk:", e)

    try:
        # 3. Gráfico Figura 3 Nature 2025 (Waterfall de Correlaciones Físicas)
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        features = [
            "Asimetria Borde", "Irregularidad Borde", "Eritema Perilesional", 
            "Asimetria Color", "Varianza Color", "Area Lesion", 
            "Diametro Menor", "Contraste Azul", "Tono Hue (Rojo)"
        ]
        corrs = [0.07, 0.18, -0.31, 0.34, 0.34, 0.35, 0.37, -0.47, -0.55]
        wf_colors = ['#ef4444' if c < 0 else '#0d9488' for c in corrs]
        
        ax.barh(features, corrs, color=wf_colors, height=0.55)
        ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
        ax.set_xlim(-0.65, 0.5)
        ax.set_xlabel("Correlacion de Spearman (rho)", fontsize=8, fontweight='bold', color='#1e293b')
        ax.set_title("Figura 3 Nature 2025: Correlaciones Fisicas Waterfall", fontsize=10, fontweight='bold', color='#0f766e')
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        waterfall_chart_path = os.path.join(output_dir, "report_waterfall_chart.png")
        plt.savefig(waterfall_chart_path)
        plt.close(fig)
    except Exception as e:
        print("Error generando chart waterfall:", e)

    try:
        # 4. Gráfico Figura 1a Nature 2025 (Scatter Plot Concordancia)
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
        pub_pauc = [0.125, 0.138, 0.145, 0.152, 0.161, 0.170, 0.178, 0.185, 0.130, 0.142, 0.158, 0.174]
        priv_pauc = [0.121, 0.134, 0.141, 0.149, 0.158, 0.166, 0.173, 0.181, 0.127, 0.139, 0.154, 0.170]
        
        ax.scatter(pub_pauc, priv_pauc, color='#0d9488', alpha=0.8, edgecolors='none', s=40)
        ax.plot([0.11, 0.19], [0.11, 0.19], color='#64748b', linestyle='--', linewidth=1, label='Bisectriz r = 1.0')
        ax.set_xlim(0.11, 0.19)
        ax.set_ylim(0.11, 0.19)
        ax.set_xlabel("Public Leaderboard pAUC", fontsize=8, fontweight='bold', color='#1e293b')
        ax.set_ylabel("Private Leaderboard pAUC", fontsize=8, fontweight='bold', color='#1e293b')
        ax.set_title("Figura 1a Nature 2025: Concordancia Public vs Private (r = 0.988)", fontsize=10, fontweight='bold', color='#0f766e')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        concordance_chart_path = os.path.join(output_dir, "report_concordance_chart.png")
        plt.savefig(concordance_chart_path)
        plt.close(fig)
    except Exception as e:
        print("Error generando chart concordance:", e)

    try:
        # 5. Gráfico Estudio de Ablación (Nature 2025 Table 3)
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
        variants = ["Modelo Completo", "Sin Contexto", "Sin WB360", "Solo Metadata", "Solo Imagenes"]
        auc_vals = [0.967, 0.956, 0.948, 0.939, 0.922]
        
        bars = ax.bar(variants, auc_vals, color="#0d9488", width=0.5)
        ax.set_ylim(0.90, 0.98)
        ax.set_ylabel("AUC Score", fontsize=9, fontweight='bold', color='#1e293b')
        ax.set_title("Estudio de Ablacion de Fuentes de Informacion (Nature 2025)", fontsize=10, fontweight='bold', color='#0f766e')
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
                        
        plt.tight_layout()
        ablation_chart_path = os.path.join(output_dir, "report_ablation_chart.png")
        plt.savefig(ablation_chart_path)
        plt.close(fig)
    except Exception as e:
        print("Error generando chart ablation:", e)

    try:
        # 6. Gráfico Ensamble Multimodal (ISIC 2024 1st Place Gain)
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
        ens_models = ["Solo Vision (CNN)", "Solo Tabular (GBDT)", "Ensamble Multimodal"]
        pauc_vals = [0.165, 0.158, 0.198]
        ens_colors = ["#3b82f6", "#64748b", "#0d9488"]
        
        bars = ax.bar(ens_models, pauc_vals, color=ens_colors, width=0.45)
        ax.set_ylim(0.12, 0.22)
        ax.set_ylabel("pAUC >80% TPR", fontsize=9, fontweight='bold', color='#1e293b')
        ax.set_title("Ganancia pAUC del Ensamble Multimodal (ISIC 2024 1st Place)", fontsize=10, fontweight='bold', color='#0f766e')
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
        plt.tight_layout()
        ensemble_chart_path = os.path.join(output_dir, "report_ensemble_chart.png")
        plt.savefig(ensemble_chart_path)
        plt.close(fig)
    except Exception as e:
        print("Error generando chart ensemble:", e)

    return {
        "mcc_chart": mcc_chart_path,
        "patient_risk_chart": patient_risk_chart_path,
        "waterfall_chart": waterfall_chart_path,
        "concordance_chart": concordance_chart_path,
        "ablation_chart": ablation_chart_path,
        "ensemble_chart": ensemble_chart_path
    }
