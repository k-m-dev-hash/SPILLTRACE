from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = BACKEND_ROOT / "models" / "spilltrace_classifier.pth"

IMAGE_SIZE = 256

CLASS_NAMES = [
    "OIL",
    "LOOKALIKE",
    "NO OIL",
]

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class SARClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def load_classifier():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found:\n{MODEL_PATH}"
        )

    model = SARClassifier(num_classes=3)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    _model = model

    print(f"SAR classifier loaded: {MODEL_PATH}")

    return _model


# ============================================================
# LOAD SENTINEL-1 TIFF
# ============================================================

def load_sar_image(image_path):
    """
    Loads a Sentinel-1 VV/VH TIFF and prepares it
    exactly like the classifier training pipeline.
    """

    image = tifffile.imread(str(image_path))

    image = np.asarray(image, dtype=np.float32)

    # Expected format: H x W x 2
    if image.ndim == 3 and image.shape[-1] == 2:
        vv = image[:, :, 0]
        vh = image[:, :, 1]

    # Also support: 2 x H x W
    elif image.ndim == 3 and image.shape[0] == 2:
        vv = image[0]
        vh = image[1]

    else:
        raise ValueError(
            f"Expected a 2-channel VV/VH TIFF, got shape {image.shape}"
        )

    # --------------------------------------------------------
    # Z-score normalization
    # Same preprocessing used during classifier training.
    # --------------------------------------------------------

    vv = (vv - vv.mean()) / (vv.std() + 1e-6)
    vh = (vh - vh.mean()) / (vh.std() + 1e-6)

    # Resize to 256 x 256
    vv = cv2.resize(
        vv,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    vh = cv2.resize(
        vh,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # [2, H, W]
    image = np.stack([vv, vh], axis=0)

    # [1, 2, H, W]
    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor


# ============================================================
# CLASSIFY SAR IMAGE
# ============================================================

def classify_sar_image(image_path):
    """
    Classifies a Sentinel-1 SAR image into:

        OIL
        LOOKALIKE
        NO OIL

    Returns probabilities and the predicted class.
    """

    model = load_classifier()

    tensor = load_sar_image(image_path)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = CLASS_NAMES[predicted_index]

    class_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i].item())
        for i in range(len(CLASS_NAMES))
    }

    confidence = float(
        probabilities[predicted_index].item()
    )

    return {
        "classification": predicted_class,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SPILLTRACE SAR CLASSIFIER TEST")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    load_classifier()

    print("\nClassifier loaded successfully.")
    print("Classes:")

    for class_name in CLASS_NAMES:
        print(f"  - {class_name}")

    print("\nReady for Sentinel-1 TIFF classification.")