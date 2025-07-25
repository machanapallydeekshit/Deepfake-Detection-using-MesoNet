import tensorflow as tf
import numpy as np
import cv2

# ----------------------------------
# Standard Grad-CAM
# ----------------------------------
def get_gradcam_heatmap(model, img_array, last_conv_layer_name='conv_last', class_index=0):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    :param model: A trained Keras model (like Meso4.model).
    :param img_array: A preprocessed (1, 256, 256, 3) numpy array.
    :param last_conv_layer_name: Name of the last Conv2D layer in the model.
    :param class_index: Index of the class to inspect. 0 if a single sigmoid output.
    :return: A 2D numpy array (the heatmap) with values in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Watch gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    # Gradients of the output wrt conv outputs
    grads = tape.gradient(loss, conv_outputs)
    # Global average pooling on the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted conv outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        max_val = tf.constant(1e-10)
    heatmap /= max_val

    return heatmap.numpy()


def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap on the original image.
    :param img: Original (H, W, 3) image in BGR or RGB (be consistent).
    :param heatmap: 2D array with values in [0,1].
    :param alpha: Heatmap intensity factor.
    :param colormap: OpenCV colormap to use, default is JET.
    :return: The resulting overlay (H, W, 3).
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    output = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return output

# ----------------------------------
# Global Metrics (Full Heatmap)
# ----------------------------------
def compute_heatmap_metrics(heatmap, threshold=0.5):
    """
    Compute global metrics for the full heatmap.
    :param heatmap: 2D numpy array, in [0,1].
    :param threshold: Float, threshold for "hot" pixels.
    :return: Dictionary with 'mean', 'max', 'fraction_above_threshold'
    """
    mean_val = np.mean(heatmap)
    max_val = np.max(heatmap)
    frac_above = np.sum(heatmap > threshold) / heatmap.size
    return {
        'mean': mean_val,
        'max': max_val,
        'fraction_above_threshold': frac_above
    }

# ----------------------------------
# Regional Metrics (Specific Regions)
# ----------------------------------

def compute_region_metrics(heatmap, roi, threshold=0.5):
    """
    Compute metrics for a specific region of interest (ROI) in the heatmap.
    :param heatmap: 2D numpy array (the Grad-CAM heatmap, normalized to [0, 1]).
    :param roi: Tuple (x1, y1, x2, y2), defining the ROI in absolute pixel coordinates.
    :param threshold: Float, used to compute fraction above threshold.
    :return: Dictionary with 'mean', 'max', 'fraction_above_threshold' for the ROI.
    """
    x1, y1, x2, y2 = roi

    # Ensure the ROI is valid
    if x1 < 0 or y1 < 0 or x2 > heatmap.shape[1] or y2 > heatmap.shape[0]:
        print(f"Invalid ROI: {roi}, heatmap size: {heatmap.shape}")
        return {"mean": 0, "max": 0, "fraction_above_threshold": 0}

    # Extract the region from the heatmap
    region = heatmap[y1:y2, x1:x2]

    # Check for empty region
    if region.size == 0:
        print(f"Empty region for ROI: {roi}")
        return {"mean": 0, "max": 0, "fraction_above_threshold": 0}

    # Compute metrics for the region
    mean_val = np.mean(region)
    max_val = np.max(region)

    # Avoid division by zero
    if region.size == 0:
        fraction_above = 0
    else:
        fraction_above = np.sum(region > threshold) / region.size

    return {
        "mean": mean_val,
        "max": max_val,
        "fraction_above_threshold": fraction_above
    }

