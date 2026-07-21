"""
pdf_generator_optimized.py
Generación de reportes PDF con los resultados del diagnóstico.
"""

import io
import os
from datetime import datetime
from typing import Optional

import streamlit as st

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, HRFlowable,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def _fig_to_bytes(fig) -> Optional[bytes]:
    """Convierte una figura matplotlib a bytes PNG."""
    if fig is None or not _MATPLOTLIB_AVAILABLE:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def _pil_to_bytes(pil_image) -> Optional[bytes]:
    """Convierte una imagen PIL a bytes JPEG."""
    try:
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def generate_pdf_report(
    image,
    diagnosis: str,
    confidence_percent: float,
    raw_confidence: float,
    model_name: str,
    model_info: dict,
    comparison_results: list,
    translations: dict,
    confidence_threshold: int,
    metrics_data: dict,
    plots_data: dict,
):
    """
    Genera un reporte PDF y lo ofrece como descarga en Streamlit.

    plots_data esperado (todos opcionales):
      {
        'confusion_matrix':  <fig o None>,
        'metrics_dashboard': <fig o None>,
        'advanced_dashboard': <fig o None>,
        'comparison_plots': {
            'Comparacion de Confianza': <fig o None>,
            ...
        }
      }
    """
    if not _REPORTLAB_AVAILABLE:
        st.error(
            "La librería reportlab no está instalada. "
            "Instálala con: pip install reportlab"
        )
        return

    t = translations
    buffer = io.BytesIO()

    try:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1976D2"),
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#424242"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = styles["Normal"]
        body_style.fontSize = 10

        story = []

        # --- Título ---
        story.append(Paragraph("🔬 DermAI — Reporte de Diagnóstico", title_style))
        story.append(Paragraph(
            f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            body_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.3 * cm))

        # --- Imagen original ---
        img_bytes = _pil_to_bytes(image)
        if img_bytes:
            story.append(Paragraph("Imagen Analizada", heading_style))
            story.append(RLImage(io.BytesIO(img_bytes), width=6 * cm, height=6 * cm))
            story.append(Spacer(1, 0.3 * cm))

        # --- Diagnóstico ---
        story.append(Paragraph("Resultado del Diagnóstico", heading_style))
        diag_color = "#F44336" if diagnosis == "Maligno" else "#4CAF50"
        story.append(Paragraph(
            f'<font color="{diag_color}"><b>{diagnosis}</b></font> — '
            f'Confianza: <b>{confidence_percent:.1f}%</b> (score raw: {raw_confidence:.4f})',
            body_style,
        ))
        story.append(Paragraph(f"Modelo utilizado: <b>{model_name}</b>", body_style))
        story.append(Spacer(1, 0.3 * cm))

        # --- Métricas del modelo ---
        if metrics_data:
            story.append(Paragraph("Métricas del Modelo (ISIC 2019)", heading_style))
            table_data = [["Métrica", "Valor"]] + [
                [k.replace("_", " ").title(), f"{v*100:.1f}%" if isinstance(v, float) and v <= 1 else str(v)]
                for k, v in metrics_data.items()
                if k != "confusion_matrix"
            ]
            tbl = Table(table_data, colWidths=[8 * cm, 8 * cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))

        # --- Comparación de modelos ---
        if comparison_results:
            story.append(Paragraph("Comparación entre Modelos", heading_style))
            comp_headers = ["Modelo", "Diagnóstico", "Confianza (%)", "Tiempo (ms)"]
            comp_rows = [comp_headers] + [
                [
                    r.get("Modelo", ""),
                    r.get("Diagnóstico", ""),
                    str(r.get("Confianza (%)", "")),
                    str(r.get("Tiempo (ms)", "")),
                ]
                for r in comparison_results
            ]
            comp_tbl = Table(comp_rows, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
            comp_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECEFF1")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(comp_tbl)
            story.append(Spacer(1, 0.3 * cm))

        # --- Gráficos ---
        for key in ("confusion_matrix", "metrics_dashboard", "advanced_dashboard"):
            fig = plots_data.get(key)
            if fig is not None:
                fig_bytes = _fig_to_bytes(fig)
                if fig_bytes:
                    story.append(Paragraph(key.replace("_", " ").title(), heading_style))
                    story.append(RLImage(io.BytesIO(fig_bytes), width=14 * cm, height=7 * cm))
                    story.append(Spacer(1, 0.2 * cm))

        for label, fig in (plots_data.get("comparison_plots") or {}).items():
            if fig is not None:
                fig_bytes = _fig_to_bytes(fig)
                if fig_bytes:
                    story.append(Paragraph(label, heading_style))
                    story.append(RLImage(io.BytesIO(fig_bytes), width=14 * cm, height=6 * cm))
                    story.append(Spacer(1, 0.2 * cm))

        # --- Disclaimer ---
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            t.get(
                "disclaimer_body",
                "⚠️ Esta herramienta es únicamente para fines educativos y de investigación. "
                "No reemplaza el diagnóstico médico profesional.",
            ),
            ParagraphStyle("Disclaimer", parent=body_style, textColor=colors.HexColor("#B71C1C"), fontSize=9),
        ))

        doc.build(story)
        pdf_bytes = buffer.getvalue()

        st.download_button(
            label="📥 Descargar Reporte PDF",
            data=pdf_bytes,
            file_name=f"dermai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error generando el PDF: {e}")
    finally:
        buffer.close()
