"""
visualization.py
Funciones de visualización: matrices de confusión, dashboards y gráficos comparativos.

Fix #5: todas las funciones devuelven la figura sin mostrarla. El llamador
es responsable de llamar st.pyplot(fig) y luego plt.close(fig).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Paleta de colores consistente
# ---------------------------------------------------------------------------
_COLORS = {
    "primary":   "#2196F3",
    "secondary": "#FF9800",
    "success":   "#4CAF50",
    "danger":    "#F44336",
    "purple":    "#9C27B0",
    "teal":      "#009688",
}


# ---------------------------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, model_name: str = ""):
    """
    Dibuja una matriz de confusión 2×2 y retorna la figura.

    Args:
        cm: lista/array [[TN, FP], [FN, TP]]
        model_name: nombre del modelo (para el título)

    Returns:
        matplotlib.figure.Figure o None si cm es None/inválido
    """
    if cm is None:
        return None
    try:
        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)

        labels = ["Benigno", "Maligno"]
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold",
                )

        ax.set_xlabel("Predicción", fontsize=12)
        ax.set_ylabel("Real", fontsize=12)
        title = f"Matriz de Confusión\n{model_name}" if model_name else "Matriz de Confusión"
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dashboard de métricas
# ---------------------------------------------------------------------------
def create_metrics_dashboard(metrics: dict, model_name: str = ""):
    """
    Crea un dashboard con barras de las métricas principales.
    Retorna la figura.
    """
    if not metrics:
        return None
    try:
        metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_score"]
        labels = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"]
        values = [metrics.get(k, 0) * 100 for k in metric_keys]
        colors = [_COLORS["primary"], _COLORS["success"], _COLORS["teal"],
                  _COLORS["secondary"], _COLORS["purple"]]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Porcentaje (%)", fontsize=11)
        ax.set_title(f"Métricas de Rendimiento — {model_name}", fontsize=13, fontweight="bold")

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dashboard avanzado con MCC y AUC
# ---------------------------------------------------------------------------
def create_advanced_metrics_dashboard(metrics: dict, model_name: str = ""):
    """
    Dashboard avanzado que incluye MCC y AUC además de las métricas estándar.
    """
    if not metrics:
        return None
    try:
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig, wspace=0.35)

        # Izquierda: métricas estándar
        ax1 = fig.add_subplot(gs[0, 0])
        metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_score"]
        labels = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"]
        values = [metrics.get(k, 0) * 100 for k in metric_keys]
        bars = ax1.barh(labels, values,
                        color=[_COLORS["primary"], _COLORS["success"], _COLORS["teal"],
                               _COLORS["secondary"], _COLORS["purple"]],
                        edgecolor="white", height=0.6)
        ax1.set_xlim(0, 115)
        ax1.set_title("Métricas Estándar (%)", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, values):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

        # Derecha: MCC y AUC
        ax2 = fig.add_subplot(gs[0, 1])
        adv_labels = ["MCC", "AUC-ROC"]
        adv_values = [metrics.get("mcc", 0), metrics.get("auc", 0)]
        adv_colors = [_COLORS["danger"], _COLORS["primary"]]
        bars2 = ax2.bar(adv_labels, adv_values, color=adv_colors, edgecolor="white", width=0.4)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Métricas Avanzadas", fontsize=12, fontweight="bold")
        for bar, val in zip(bars2, adv_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

        fig.suptitle(f"Dashboard Avanzado — {model_name}", fontsize=13, fontweight="bold")
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gráficos de comparación de modelos
# ---------------------------------------------------------------------------
def create_model_comparison_plots(df_comparison):
    """
    Crea dos gráficas: confianza y tiempo de inferencia por modelo.
    Retorna (fig_confianza, fig_tiempo).
    """
    if df_comparison is None or df_comparison.empty:
        return None, None

    try:
        models = df_comparison["Modelo"].tolist()
        confidences = df_comparison["Confianza (%)"].tolist()
        times = df_comparison["Tiempo (ms)"].tolist()

        # --- Confianza ---
        fig_conf, ax_conf = plt.subplots(figsize=(8, 3.5))
        colors_conf = [
            _COLORS["danger"] if d == "Maligno" else _COLORS["success"]
            for d in df_comparison["Diagnóstico"].tolist()
        ]
        ax_conf.bar(models, confidences, color=colors_conf, edgecolor="white")
        ax_conf.set_ylim(0, 110)
        ax_conf.set_ylabel("Confianza (%)")
        ax_conf.set_title("Comparación de Confianza por Modelo", fontweight="bold")
        for i, (m, v) in enumerate(zip(models, confidences)):
            ax_conf.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
        patches = [
            mpatches.Patch(color=_COLORS["danger"], label="Maligno"),
            mpatches.Patch(color=_COLORS["success"], label="Benigno"),
        ]
        ax_conf.legend(handles=patches, loc="upper right")
        fig_conf.tight_layout()

        # --- Tiempo ---
        fig_time, ax_time = plt.subplots(figsize=(8, 3.5))
        ax_time.bar(models, times, color=_COLORS["primary"], edgecolor="white")
        ax_time.set_ylabel("Tiempo (ms)")
        ax_time.set_title("Velocidad de Inferencia por Modelo", fontweight="bold")
        for i, (m, v) in enumerate(zip(models, times)):
            ax_time.text(i, v + 0.5, f"{v:.1f} ms", ha="center", fontsize=9)
        fig_time.tight_layout()

        return fig_conf, fig_time
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Gráfico comparativo de MCC
# ---------------------------------------------------------------------------
def create_mcc_comparison_chart(mcc_data: list[dict]):
    """
    Barra horizontal de MCC para todos los modelos.
    """
    if not mcc_data:
        return None
    try:
        models = [d["model"] for d in mcc_data]
        mccs = [d["mcc"] for d in mcc_data]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = [_COLORS["primary"] if v >= 0.75 else
                  _COLORS["success"] if v >= 0.6 else
                  _COLORS["secondary"] for v in mccs]
        bars = ax.barh(models, mccs, color=colors, edgecolor="white", height=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("MCC")
        ax.set_title("Comparación MCC entre Modelos", fontweight="bold")
        for bar, val in zip(bars, mccs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gráfico de p-valores de McNemar
# ---------------------------------------------------------------------------
def create_mcnemar_plot(mcnemar_results: list[dict]):
    """
    Diagrama de barras de p-valores de las pruebas de McNemar.
    Línea roja discontinua en α=0.05.
    """
    if not mcnemar_results:
        return None
    try:
        pairs = [f"{r['model_a']}\nvs\n{r['model_b']}" for r in mcnemar_results]
        pvals = [r["p_value"] for r in mcnemar_results]
        colors = [_COLORS["danger"] if p < 0.05 else _COLORS["primary"] for p in pvals]

        fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.5), 4))
        ax.bar(range(len(pairs)), pvals, color=colors, edgecolor="white")
        ax.axhline(0.05, color="red", linestyle="--", label="α = 0.05")
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, fontsize=8)
        ax.set_ylabel("p-valor")
        ax.set_title("Pruebas de McNemar — p-valores", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception:
        return None
