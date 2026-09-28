import os
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

from modules.spill_detection import detect_dark_regions


# ============================================================
# SPILLTRACE - DATASET EVALUATION
# ============================================================

DATASET_ROOT = Path(
    os.path.expanduser(
        r"~\Downloads\02_Test_images_and_ground_truth"
    )
)

IMAGES_ROOT = DATASET_ROOT / "Images"
MASK_ROOT = DATASET_ROOT / "Mask"

OUTPUT_FILE = Path("evaluation_results.json")


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================

def load_ground_truth(mask_path):

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # Ground truth contains 0 and 1
    return (mask > 0).astype(np.uint8)


# ============================================================
# CREATE PREDICTION MASK FROM DETECTED CANDIDATES
# ============================================================

def create_prediction_mask(detection):

    height = detection["image_height"]
    width = detection["image_width"]

    prediction = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    candidates = detection.get("candidates", [])

    for candidate in candidates:

        x = int(candidate["centroid_x"])
        y = int(candidate["centroid_y"])

        candidate_width = int(
            candidate["width_pixels"]
        )

        candidate_height = int(
            candidate["height_pixels"]
        )

        # Reconstruct approximate bounding box
        x1 = max(
            0,
            x - candidate_width // 2
        )

        y1 = max(
            0,
            y - candidate_height // 2
        )

        x2 = min(
            width - 1,
            x + candidate_width // 2
        )

        y2 = min(
            height - 1,
            y + candidate_height // 2
        )

        cv2.rectangle(
            prediction,
            (x1, y1),
            (x2, y2),
            1,
            -1
        )

    return prediction


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    true_positive = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    false_positive = np.logical_and(
        prediction,
        np.logical_not(ground_truth)
    ).sum()

    false_negative = np.logical_and(
        np.logical_not(prediction),
        ground_truth
    ).sum()

    true_negative = np.logical_and(
        np.logical_not(prediction),
        np.logical_not(ground_truth)
    ).sum()

    # Precision
    precision = (
        true_positive /
        (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    # Recall
    recall = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    # F1
    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # IoU
    intersection = true_positive

    union = (
        true_positive +
        false_positive +
        false_negative
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    # False Positive Rate
    false_positive_rate = (
        false_positive /
        (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    return {
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),

        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "iou": float(iou),
        "false_positive_rate": float(
            false_positive_rate
        )
    }


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def evaluate_image(
    image_path,
    mask_path
):

    detection = detect_dark_regions(
        str(image_path)
    )

    prediction = create_prediction_mask(
        detection
    )

    ground_truth = load_ground_truth(
        mask_path
    )

    if prediction.shape != ground_truth.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"prediction={prediction.shape}, "
            f"ground_truth={ground_truth.shape}"
        )

    metrics = calculate_metrics(
        prediction,
        ground_truth
    )

    return {
        "filename": image_path.name,
        "metrics": metrics,
        "candidate_count": detection[
            "candidate_count"
        ]
    }


# ============================================================
# PROCESS CATEGORY
# ============================================================

def process_category(category):

    image_folder = IMAGES_ROOT / category
    mask_folder = MASK_ROOT / category

    results = []

    image_files = sorted(
        image_folder.glob("*.tif")
    )

    print()
    print("=" * 70)
    print(f"CATEGORY: {category}")
    print("=" * 70)

    print(
        f"Images found: {len(image_files)}"
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # 00000.tif -> 00000_segmentation.tif

        mask_name = (
            image_path.stem +
            "_segmentation.tif"
        )

        mask_path = (
            mask_folder /
            mask_name
        )

        if not mask_path.exists():

            print(
                f"[{index}/{len(image_files)}] "
                f"MASK NOT FOUND: "
                f"{mask_name}"
            )

            continue

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        try:

            result = evaluate_image(
                image_path,
                mask_path
            )

            results.append(result)

            metrics = result["metrics"]

            print(
                f"    Candidates: "
                f"{result['candidate_count']}"
            )

            print(
                f"    IoU: "
                f"{metrics['iou']:.4f}"
            )

            print(
                f"    Precision: "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"    Recall: "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"    F1: "
                f"{metrics['f1_score']:.4f}"
            )

        except Exception as error:

            print(
                f"    ERROR: {error}"
            )

    return results


# ============================================================
# AVERAGE METRICS
# ============================================================

def calculate_average(results):

    if not results:

        return {
            "images_evaluated": 0
        }

    metric_names = [
        "precision",
        "recall",
        "f1_score",
        "iou",
        "false_positive_rate"
    ]

    averages = {}

    for metric in metric_names:

        values = [
            result["metrics"][metric]
            for result in results
        ]

        averages[metric] = float(
            np.mean(values)
        )

    averages["images_evaluated"] = len(
        results
    )

    return averages


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SPILLTRACE DATASET EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Dataset: {DATASET_ROOT}"
    )

    all_results = {}

    for category in [
        "Oil",
        "Lookalike",
        "No oil"
    ]:

        category_results = process_category(
            category
        )

        all_results[category] = {
            "results": category_results,
            "average_metrics":
                calculate_average(
                    category_results
                )
        }

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    combined_results = []

    for category_data in all_results.values():

        combined_results.extend(
            category_data["results"]
        )

    overall = calculate_average(
        combined_results
    )

    output = {
        "dataset": str(DATASET_ROOT),

        "categories": all_results,

        "overall": overall
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    for category, data in all_results.items():

        metrics = data["average_metrics"]

        print()
        print(category)

        print(
            f"  Images: "
            f"{metrics.get('images_evaluated', 0)}"
        )

        print(
            f"  Precision: "
            f"{metrics.get('precision', 0):.4f}"
        )

        print(
            f"  Recall: "
            f"{metrics.get('recall', 0):.4f}"
        )

        print(
            f"  F1: "
            f"{metrics.get('f1_score', 0):.4f}"
        )

        print(
            f"  IoU: "
            f"{metrics.get('iou', 0):.4f}"
        )

        print(
            f"  False Positive Rate: "
            f"{metrics.get('false_positive_rate', 0):.4f}"
        )

    print()
    print("OVERALL")

    print(
        f"  Images: "
        f"{overall.get('images_evaluated', 0)}"
    )

    print(
        f"  Precision: "
        f"{overall.get('precision', 0):.4f}"
    )

    print(
        f"  Recall: "
        f"{overall.get('recall', 0):.4f}"
    )

    print(
        f"  F1: "
        f"{overall.get('f1_score', 0):.4f}"
    )

    print(
        f"  IoU: "
        f"{overall.get('iou', 0):.4f}"
    )

    print(
        f"  False Positive Rate: "
        f"{overall.get('false_positive_rate', 0):.4f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_FILE.absolute()}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()