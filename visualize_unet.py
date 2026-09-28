import os
import random

import cv2
import numpy as np
import tifffile
import torch
import matplotlib.pyplot as plt

from dataset import build_samples
from model import UNet


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = r"C:\Users\Khushi Sharma\Downloads\02_Test_images_and_ground_truth"

MODEL_PATH = r"models\spilltrace_unet.pth"

OUTPUT_ROOT = r"unet_visual_analysis"

IMAGE_SIZE = 256

THRESHOLD = 0.5

SEED = 42


# ============================================================
# Setup
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")


os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# Load all samples
# ============================================================

samples = build_samples(
    DATASET_ROOT
)


# ============================================================
# Reproduce validation split
# ============================================================

categories = {}

for sample in samples:

    category = sample[2]

    if category not in categories:
        categories[category] = []

    categories[category].append(sample)


validation_samples = []


for category, category_samples in categories.items():

    random.shuffle(category_samples)

    split_index = int(
        len(category_samples) * 0.8
    )

    validation_samples.extend(
        category_samples[split_index:]
    )


# ============================================================
# Pick one example from each category
# ============================================================

selected = {}

for sample in validation_samples:

    category = sample[2]

    if category not in selected:
        selected[category] = sample


# ============================================================
# Process image
# ============================================================

def load_image(image_path):

    image = tifffile.imread(
        image_path
    )

    if image.shape[-1] == 2:

        vv = image[:, :, 0]
        vh = image[:, :, 1]

    elif image.shape[0] == 2:

        vv = image[0]
        vh = image[1]

    else:

        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )


    vv = np.nan_to_num(
        vv,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    vh = np.nan_to_num(
        vh,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # Normalize exactly like training

    vv = (
        vv - np.mean(vv)
    ) / (
        np.std(vv) + 1e-6
    )

    vh = (
        vh - np.mean(vh)
    ) / (
        np.std(vh) + 1e-6
    )


    # Resize

    vv_small = cv2.resize(
        vv,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    vh_small = cv2.resize(
        vh,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )


    model_input = np.stack(
        [vv_small, vh_small],
        axis=0
    ).astype(np.float32)


    return image, model_input


# ============================================================
# Process selected examples
# ============================================================

for category, sample in selected.items():

    image_path, mask_path, category = sample

    print()
    print(
        "Analyzing:",
        category
    )

    print(
        "Image:",
        os.path.basename(image_path)
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image, model_input = load_image(
        image_path
    )


    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = tifffile.imread(
        mask_path
    )

    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)


    ground_truth_small = cv2.resize(
        ground_truth,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        model_input
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        logits = model(
            tensor
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()


    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Create SAR display
    # --------------------------------------------------------

    vv = original_image[:, :, 0]

    vv_display = np.nan_to_num(
        vv
    )

    low, high = np.percentile(
        vv_display,
        [1, 99]
    )

    vv_display = np.clip(
        vv_display,
        low,
        high
    )

    vv_display = (
        (vv_display - low)
        /
        (high - low + 1e-8)
    )


    # --------------------------------------------------------
    # Resize SAR
    # --------------------------------------------------------

    vv_display = cv2.resize(
        vv_display,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = np.stack(
        [
            vv_display,
            vv_display,
            vv_display
        ],
        axis=-1
    )


    # Green = ground truth only
    ground_truth_only = (
        (ground_truth_small == 1)
        &
        (prediction == 0)
    )


    # Red = prediction only
    prediction_only = (
        (prediction == 1)
        &
        (ground_truth_small == 0)
    )


    # Yellow = overlap
    overlap = (
        (prediction == 1)
        &
        (ground_truth_small == 1)
    )


    overlay[ground_truth_only] = [
        0,
        1,
        0
    ]

    overlay[prediction_only] = [
        1,
        0,
        0
    ]

    overlay[overlap] = [
        1,
        1,
        0
    ]


    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    category_dir = os.path.join(
        OUTPUT_ROOT,
        category.replace(" ", "_")
    )

    os.makedirs(
        category_dir,
        exist_ok=True
    )


    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]


    # SAR

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        vv_display,
        cmap="gray"
    )

    plt.title(
        f"{category} - SAR VV"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            category_dir,
            f"{base_name}_sar.png"
        ),
        dpi=150
    )

    plt.close()


    # Ground truth

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        ground_truth_small,
        cmap="gray"
    )

    plt.title(
        f"{category} - Ground Truth"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            category_dir,
            f"{base_name}_ground_truth.png"
        ),
        dpi=150
    )

    plt.close()


    # Prediction

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        label="Oil Probability"
    )

    plt.title(
        f"{category} - U-Net Prediction"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            category_dir,
            f"{base_name}_prediction.png"
        ),
        dpi=150
    )

    plt.close()


    # Overlay

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"{category} - Prediction Overlay"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            category_dir,
            f"{base_name}_overlay.png"
        ),
        dpi=150
    )

    plt.close()


    print(
        "Saved to:",
        category_dir
    )


print()
print("=" * 60)
print("U-NET VISUAL ANALYSIS COMPLETE")
print("=" * 60)

print(
    "Results:",
    OUTPUT_ROOT
)