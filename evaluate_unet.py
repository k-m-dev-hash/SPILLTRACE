import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_samples, OilSpillDataset
from model import UNet


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = r"C:\Users\Khushi Sharma\Downloads\02_Test_images_and_ground_truth"

MODEL_PATH = r"models\spilltrace_unet.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 4

TRAIN_RATIO = 0.8
SEED = 42

THRESHOLD = 0.5


# ============================================================
# Reproduce the same validation split used during training
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")


print("=" * 60)
print("SPILLTRACE U-NET EVALUATION")
print("=" * 60)

print("Device:", device)


# ============================================================
# Load samples
# ============================================================

samples = build_samples(
    DATASET_ROOT
)

print()
print("Total samples:", len(samples))


# ============================================================
# Recreate the same stratified split
# ============================================================

categories = {}

for sample in samples:

    category = sample[2]

    if category not in categories:
        categories[category] = []

    categories[category].append(sample)


train_samples = []
val_samples = []


for category, category_samples in categories.items():

    random.shuffle(category_samples)

    split_index = int(
        len(category_samples)
        * TRAIN_RATIO
    )

    train_samples.extend(
        category_samples[:split_index]
    )

    val_samples.extend(
        category_samples[split_index:]
    )

    print(
        f"{category}: "
        f"{len(category_samples)} total | "
        f"{split_index} train | "
        f"{len(category_samples) - split_index} validation"
    )


random.shuffle(train_samples)
random.shuffle(val_samples)


print()
print("Validation samples:", len(val_samples))


# ============================================================
# Dataset
# ============================================================

val_dataset = OilSpillDataset(
    val_samples,
    image_size=IMAGE_SIZE
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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
# Metrics
# ============================================================

category_stats = {}

for category in categories:

    category_stats[category] = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "images": 0
    }


total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0


# ============================================================
# Evaluation
# ============================================================

sample_index = 0


with torch.no_grad():

    for images, masks, batch_categories in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()


        for i in range(images.size(0)):

            prediction = predictions[i, 0].cpu().numpy()
            target = masks[i, 0].cpu().numpy()

            prediction = prediction.astype(bool)
            target = target.astype(bool)

            tp = np.logical_and(
                prediction,
                target
            ).sum()

            fp = np.logical_and(
                prediction,
                np.logical_not(target)
            ).sum()

            fn = np.logical_and(
                np.logical_not(prediction),
                target
            ).sum()

            tn = np.logical_and(
                np.logical_not(prediction),
                np.logical_not(target)
            ).sum()


            category = batch_categories[i]


            category_stats[category]["tp"] += int(tp)
            category_stats[category]["fp"] += int(fp)
            category_stats[category]["fn"] += int(fn)
            category_stats[category]["tn"] += int(tn)
            category_stats[category]["images"] += 1


            total_tp += int(tp)
            total_fp += int(fp)
            total_fn += int(fn)
            total_tn += int(tn)


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(tp, fp, fn, tn):

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "false_positive_rate": false_positive_rate
    }


# ============================================================
# Print category results
# ============================================================

print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)


for category, stats in category_stats.items():

    metrics = calculate_metrics(
        stats["tp"],
        stats["fp"],
        stats["fn"],
        stats["tn"]
    )

    print()
    print(category)
    print("-" * 40)

    print(
        "Images:",
        stats["images"]
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1: {metrics['f1']:.4f}"
    )

    print(
        f"IoU: {metrics['iou']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.4f}"
    )


# ============================================================
# Overall results
# ============================================================

overall = calculate_metrics(
    total_tp,
    total_fp,
    total_fn,
    total_tn
)


print()
print("=" * 60)
print("OVERALL")
print("=" * 60)

print(
    "Validation images:",
    len(val_samples)
)

print(
    f"Precision: {overall['precision']:.4f}"
)

print(
    f"Recall: {overall['recall']:.4f}"
)

print(
    f"F1: {overall['f1']:.4f}"
)

print(
    f"IoU: {overall['iou']:.4f}"
)

print(
    f"False Positive Rate: "
    f"{overall['false_positive_rate']:.4f}"
)

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)