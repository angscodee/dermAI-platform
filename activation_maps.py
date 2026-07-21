"""
activation_maps.py
Generación de mapas de activación Grad-CAM para explicabilidad del diagnóstico.

Fix #3: antes de generar el mapa se fuerza una llamada al modelo (warm-up)
para garantizar que el grafo esté construido, evitando el error:
"The layer sequential has never been called and thus has no defined output."
"""

import numpy as np
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


def _ensure_model_built(model, input_shape):
    """
    Fuerza una pasada dummy a través del modelo para construir el grafo.
    Se llama antes de cualquier operación de Grad-CAM.
    """
    try:
        dummy = np.zeros((1, *input_shape), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception:
        try:
            model.build((None, *input_shape))
        except Exception:
            pass  # Si ambos fallan, el error se capturará más adelante


def _find_last_conv_layer(model):
    """
    Encuentra el nombre de la última capa convolucional del modelo.
    Busca hacia atrás en la lista de capas.
    """
    for layer in reversed(model.layers):
        if hasattr(layer, "filters"):
            return layer.name
    return None


def generate_gradcam(model, processed_image: np.ndarray, class_index: int = 0) -> Optional[np.ndarray]:
    """
    Genera un mapa de calor Grad-CAM para la imagen dada.

    Args:
        model: modelo Keras ya cargado.
        processed_image: array de shape (1, H, W, C) normalizado [0,1].
        class_index: índice de la clase de interés (0=benigno, 1=maligno normalmente).

    Returns:
        heatmap: np.ndarray de shape (H, W) normalizado [0, 1], o None si falla.
    """
    if not _TF_AVAILABLE:
        return None

    try:
        input_shape = processed_image.shape[1:]  # (H, W, C)
        _ensure_model_built(model, input_shape)

        # Obtener nombre de la última capa conv
        last_conv_name = _find_last_conv_layer(model)
        if last_conv_name is None:
            return None

        # Crear sub-modelo que retorna la salida de la última conv y la predicción final
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output],
        )

        # Calcular gradientes
        img_tensor = tf.cast(processed_image, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            # Para clasificación binaria con una sola neurona de salida
            if predictions.shape[-1] == 1:
                loss = predictions[:, 0]
            else:
                loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        # Pooling global de los gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Ponderar los mapas de activación
        conv_outputs = conv_outputs[0]  # (H', W', C_conv)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalizar a [0, 1]
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    except Exception as e:
        print(f"[activation_maps] Error en Grad-CAM: {e}")
        return None


def overlay_gradcam(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superpone el mapa de calor Grad-CAM sobre la imagen original.

    Args:
        original_image: array (H, W, 3) en rango [0, 255] o [0, 1].
        heatmap: array (H', W') normalizado [0, 1].
        alpha: intensidad de la superposición.

    Returns:
        Imagen RGB con el mapa superpuesto, shape (H, W, 3), dtype uint8.
    """
    try:
        import cv2
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        # Normalizar imagen original a uint8
        img = original_image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed
    except ImportError:
        # Fallback sin cv2: superposición manual con matplotlib
        from PIL import Image as PILImage
        h, w = original_image.shape[:2]
        heatmap_small = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), PILImage.LANCZOS)
        ) / 255.0
        cmap = plt.get_cmap("jet")
        heatmap_rgb = (cmap(heatmap_small)[:, :, :3] * 255).astype(np.uint8)
        img = original_image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return np.clip(img * (1 - alpha) + heatmap_rgb * alpha, 0, 255).astype(np.uint8)


def create_gradcam_figure(original_image: np.ndarray, processed_image: np.ndarray,
                           model, model_name: str = "", class_index: int = 0):
    """
    Genera una figura matplotlib con la imagen original y el mapa Grad-CAM.

    Returns:
        matplotlib.figure.Figure o None si no se puede generar.
    """
    heatmap = generate_gradcam(model, processed_image, class_index=class_index)
    if heatmap is None:
        return None

    try:
        # Imagen original en formato (H, W, 3)
        orig = np.array(original_image)
        if orig.ndim == 4:
            orig = orig[0]

        overlay = overlay_gradcam(orig, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(orig if orig.max() > 1 else (orig * 255).astype(np.uint8))
        axes[0].set_title("Imagen Original", fontweight="bold")
        axes[0].axis("off")

        im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Mapa Grad-CAM", fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Superposición", fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(f"Grad-CAM — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"[activation_maps] Error creando figura Grad-CAM: {e}")
        return None
