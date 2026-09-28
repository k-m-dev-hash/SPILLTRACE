import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import rasterio

from ml.model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = BACKEND_ROOT / "models" / "spilltrace_unet.pth"

IMAGE_SIZE = 256

# Same threshold used during current inference
THRESHOLD = 0.50

# Ignore extremely tiny disconnected predictions.
# This is a noise filter, NOT a requirement that an image
# must contain a particular number of spills.
MIN_COMPONENT_AREA = 100

DEVICE = torch.device("cpu")

_model = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"U-Net model not found: {MODEL_PATH}"
        )

    model = UNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support either a raw state_dict or a checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    _model = model

    print(f"U-Net loaded: {MODEL_PATH}")

    return _model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_sar_image(image_path):
    """
    Load Sentinel-1 SAR TIFF.

    Expected input:
        H x W x 2
    or:
        2 x H x W

    Channel 0 = VV
    Channel 1 = VH

    Returns:
        Tensor [1, 2, 256, 256]
    """

    image = tifffile.imread(image_path)

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D SAR image, got shape {image.shape}"
        )

    # Convert CHW -> HWC if necessary
    if image.shape[0] == 2 and image.shape[-1] != 2:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] != 2:
        raise ValueError(
            f"Expected 2 SAR channels (VV/VH), got shape {image.shape}"
        )

    image = image.astype(np.float32)

    processed = np.zeros_like(image, dtype=np.float32)

    # Same per-channel normalization used during training
    for c in range(2):
        channel = image[:, :, c]

        mean = np.mean(channel)
        std = np.std(channel)

        if std < 1e-6:
            std = 1.0

        processed[:, :, c] = (channel - mean) / std

    # Resize to model input size
    processed = cv2.resize(
        processed,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # HWC -> CHW
    processed = np.transpose(processed, (2, 0, 1))

    tensor = torch.from_numpy(processed).float()

    return tensor.unsqueeze(0).to(DEVICE)


# ============================================================
# GEOLOCATION
# ============================================================

def pixel_to_latlon(image_path, x, y):
    """
    Convert original image pixel coordinates to latitude/longitude.

    Because the model operates at 256x256, x/y must be converted
    back to the original raster dimensions before applying the
    GeoTIFF transform.
    """

    with rasterio.open(image_path) as src:

        original_width = src.width
        original_height = src.height

        scale_x = original_width / IMAGE_SIZE
        scale_y = original_height / IMAGE_SIZE

        original_x = x * scale_x
        original_y = y * scale_y

        lon, lat = rasterio.transform.xy(
            src.transform,
            original_y,
            original_x,
            offset="center"
        )

    return {
        "latitude": float(lat),
        "longitude": float(lon)
    }


# ============================================================
# MASK CLEANING
# ============================================================

def clean_prediction(mask):
    """
    Remove small isolated noise and close small gaps.
    """

    mask = mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    # Remove isolated noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Close small holes/gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return mask


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def extract_components(mask, probability_mask):
    """
    Extract every meaningful connected spill region.

    Returns a list of components instead of keeping only
    the largest contour.
    """

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    # Label 0 = background
    for label in range(1, num_labels):

        area = int(stats[label, cv2.CC_STAT_AREA])

        # Remove tiny disconnected noise
        if area < MIN_COMPONENT_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])

        component_mask = labels == label

        probabilities = probability_mask[component_mask]

        if probabilities.size > 0:
            mean_confidence = float(np.mean(probabilities))
            max_confidence = float(np.max(probabilities))
        else:
            mean_confidence = 0.0
            max_confidence = 0.0

        components.append({
            "component_label": int(label),
            "area_pixels": area,
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "centroid": {
                "x": centroid_x,
                "y": centroid_y
            },
            "mean_confidence": mean_confidence,
            "max_confidence": max_confidence
        })

    # Largest first
    components.sort(
        key=lambda item: item["area_pixels"],
        reverse=True
    )

    # Give stable candidate numbers
    for index, component in enumerate(components, start=1):
        component["candidate_number"] = index

    return components


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_spill_unet(image_path):
    """
    Run U-Net spill segmentation.

    The function detects ANY number of meaningful connected
    regions in the uploaded image.

    It does NOT assume there is exactly one spill.
    """

    model = load_model()

    image_path = str(image_path)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    input_tensor = load_sar_image(image_path)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(output)

        probability_mask = (
            probability[0, 0]
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    binary_mask = (
        probability_mask >= THRESHOLD
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    binary_mask = clean_prediction(binary_mask)

    # --------------------------------------------------------
    # Extract ALL components
    # --------------------------------------------------------

    components = extract_components(
        binary_mask,
        probability_mask
    )

    # --------------------------------------------------------
    # Add geographic coordinates
    # --------------------------------------------------------

    for component in components:

        centroid = component["centroid"]

        geo = pixel_to_latlon(
            image_path,
            centroid["x"],
            centroid["y"]
        )

        component["geographic_location"] = geo

    # --------------------------------------------------------
    # Overall detection confidence
    # --------------------------------------------------------

    detected_pixels = probability_mask[
        binary_mask > 0
    ]

    if detected_pixels.size > 0:

        mean_confidence = float(
            np.mean(detected_pixels)
        )

        max_confidence = float(
            np.max(detected_pixels)
        )

    else:

        mean_confidence = 0.0
        max_confidence = 0.0

    # --------------------------------------------------------
    # Backward-compatible primary geometry
    # --------------------------------------------------------

    if components:

        primary = components[0]

        geometry = {
            "detected": True,
            "area_pixels": primary["area_pixels"],
            "largest_contour_area_pixels": primary["area_pixels"],
            "bbox": primary["bbox"],
            "centroid": primary["centroid"],
            "contour_count": len(components)
        }

        geographic_location = primary[
            "geographic_location"
        ]

    else:

        geometry = {
            "detected": False,
            "area_pixels": 0,
            "largest_contour_area_pixels": 0,
            "bbox": None,
            "centroid": None,
            "contour_count": 0
        }

        geographic_location = None

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "detector": "SPILLTRACE U-Net",
        "model": MODEL_PATH.name,
        "threshold": THRESHOLD,

        "spill_detected": len(components) > 0,

        "detection_count": len(components),

        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence,

        # Primary / largest region for backward compatibility
        "geometry": geometry,
        "geographic_location": geographic_location,

        # NEW: every detected region
        "detections": components,

        # Full masks
        "mask": binary_mask,
        "probability_mask": probability_mask
    }


# ============================================================
# SAVE MASK
# ============================================================

def save_prediction_mask(mask, output_path):
    """
    Save binary prediction mask as PNG.
    """

    mask = np.asarray(mask)

    mask = (
        (mask > 0).astype(np.uint8) * 255
    )

    cv2.imwrite(
        str(output_path),
        mask
    )