import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from modules.spill_detection import detect_dark_regions


# ============================================================
# SPILLTRACE - VISUAL ERROR ANALYSIS
# ============================================================

DATASET_ROOT = Path(
    os.path.expanduser(
        r"~\Downloads\02_Test_images_and_ground_truth"
    )
)

IMAGES_ROOT = DATASET_ROOT / "Images"
MASK_ROOT = DATASET_ROOT / "Mask"

OUTPUT_ROOT = Path("visual_analysis")

OUTPUT_ROOT.mkdir(
    exist_ok=True
)


# ============================================================
# LOAD SAR IMAGE
# ============================================================

def load_sar_image(image_path):

    with rasterio.open(image_path) as src:

        data = src.read()

    # Expected shape:
    # (2, 2048, 2048)
    # or
    # (2048, 2048, 2)

    if data.ndim != 3:
        raise ValueError(
            f"Unexpected image shape: {data.shape}"
        )

    if data.shape[0] == 2:

        vv = data[0]

    elif data.shape[-1] == 2:

        vv = data[:, :, 0]

    else:

        raise ValueError(
            f"Could not identify VV channel: {data.shape}"
        )

    vv = np.nan_to_num(
        vv,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Robust visualization normalization

    low = np.percentile(vv, 1)
    high = np.percentile(vv, 99)

    image = np.clip(
        (vv - low) / (high - low + 1e-8),
        0,
        1
    )

    return image


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_mask(mask_path):

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    return mask > 0


# ============================================================
# CREATE PREDICTION MASK
# ============================================================

def create_prediction_mask(
    detection,
    height,
    width
):

    prediction = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for candidate in detection["candidates"]:

        cx = int(
            candidate["centroid_x"]
        )

        cy = int(
            candidate["centroid_y"]
        )

        w = int(
            candidate["width_pixels"]
        )

        h = int(
            candidate["height_pixels"]
        )

        x1 = max(
            0,
            cx - w // 2
        )

        y1 = max(
            0,
            cy - h // 2
        )

        x2 = min(
            width - 1,
            cx + w // 2
        )

        y2 = min(
            height - 1,
            cy + h // 2
        )

        cv2.rectangle(
            prediction,
            (x1, y1),
            (x2, y2),
            1,
            -1
        )

    return prediction.astype(bool)


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    sar,
    ground_truth,
    prediction
):

    # Convert SAR to RGB

    rgb = np.stack(
        [sar, sar, sar],
        axis=-1
    )

    overlay = rgb.copy()

    # Ground truth:
    # green

    gt_only = np.logical_and(
        ground_truth,
        np.logical_not(prediction)
    )

    # Prediction:
    # red

    pred_only = np.logical_and(
        prediction,
        np.logical_not(ground_truth)
    )

    # Correct overlap:
    # yellow

    overlap = np.logical_and(
        ground_truth,
        prediction
    )

    overlay[gt_only] = [
        0.0,
        1.0,
        0.0
    ]

    overlay[pred_only] = [
        1.0,
        0.0,
        0.0
    ]

    overlay[overlap] = [
        1.0,
        1.0,
        0.0
    ]

    return overlay


# ============================================================
# SAVE ONE ANALYSIS
# ============================================================

def analyze_image(
    category,
    image_number
):

    image_name = (
        f"{image_number:05d}.tif"
    )

    mask_name = (
        f"{image_number:05d}"
        "_segmentation.tif"
    )

    image_path = (
        IMAGES_ROOT /
        category /
        image_name
    )

    mask_path = (
        MASK_ROOT /
        category /
        mask_name
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    print(
        f"Analyzing {category}/{image_name}"
    )

    # Load image

    sar = load_sar_image(
        image_path
    )

    height, width = sar.shape

    # Ground truth

    ground_truth = load_mask(
        mask_path
    )

    # Detection

    detection = detect_dark_regions(
        str(image_path)
    )

    # Prediction

    prediction = create_prediction_mask(
        detection,
        height,
        width
    )

    # Overlay

    overlay = create_overlay(
        sar,
        ground_truth,
        prediction
    )

    # --------------------------------------------------------
    # Calculate metrics for this image
    # --------------------------------------------------------

    tp = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(ground_truth)
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        ground_truth
    ).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    # --------------------------------------------------------
    # Save three separate images
    # --------------------------------------------------------

    category_folder = (
        OUTPUT_ROOT / category
    )

    category_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    sar_path = (
        category_folder /
        f"{image_number:05d}_sar.png"
    )

    gt_path = (
        category_folder /
        f"{image_number:05d}_ground_truth.png"
    )

    overlay_path = (
        category_folder /
        f"{image_number:05d}_overlay.png"
    )

    plt.imsave(
        sar_path,
        sar,
        cmap="gray"
    )

    plt.imsave(
        gt_path,
        ground_truth,
        cmap="gray"
    )

    plt.imsave(
        overlay_path,
        overlay
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("Candidates:", detection["candidate_count"])

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"IoU: {iou:.4f}"
    )

    print()
    print("Saved:")
    print(sar_path)
    print(gt_path)
    print(overlay_path)

    return {
        "category": category,
        "image": image_name,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "candidate_count":
            detection["candidate_count"]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SPILLTRACE VISUAL ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Representative images
    # --------------------------------------------------------
    #
    # We intentionally use a few images instead of all 450.
    #
    # Oil:
    #     00000
    #
    # Lookalike:
    #     00000
    #
    # No oil:
    #     00000
    #
    # --------------------------------------------------------

    test_cases = [
        ("Oil", 0),
        ("Lookalike", 0),
        ("No oil", 0)
    ]

    results = []

    for category, image_number in test_cases:

        try:

            result = analyze_image(
                category,
                image_number
            )

            results.append(result)

        except Exception as error:

            print()
            print(
                f"ERROR: {category} "
                f"{image_number:05d}"
            )

            print(error)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_ROOT /
        "summary.json"
    )

    import json

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print("=" * 70)
    print("VISUAL ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results folder: "
        f"{OUTPUT_ROOT.absolute()}"
    )


if __name__ == "__main__":
    main()