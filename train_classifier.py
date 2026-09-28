from pathlib import Path
import random

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\Khushi Sharma\Downloads\02_Test_images_and_ground_truth"
)

IMAGE_ROOT = DATASET_ROOT / "Images"

MODEL_OUTPUT = Path(
    r"C:\SPILLTRACE\backend\models\spilltrace_classifier.pth"
)

IMAGE_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 0.001

SEED = 42


CLASS_NAMES = [
    "Oil",
    "Lookalike",
    "No oil",
]

CLASS_TO_INDEX = {
    "Oil": 0,
    "Lookalike": 1,
    "No oil": 2,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# LOAD SAR IMAGE
# ============================================================

def load_sar_image(path):

    image = tifffile.imread(str(path))

    if image.ndim != 3:
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    # H x W x 2
    if image.shape[-1] == 2:

        vv = image[:, :, 0]
        vh = image[:, :, 1]

    # 2 x H x W
    elif image.shape[0] == 2:

        vv = image[0]
        vh = image[1]

    else:

        raise ValueError(
            f"Could not identify VV/VH channels: "
            f"{image.shape}"
        )

    vv = vv.astype(np.float32)
    vh = vh.astype(np.float32)

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

    # Same normalization style as U-Net
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

    image = np.stack(
        [vv, vh],
        axis=0
    )

    return image.astype(np.float32)


# ============================================================
# BUILD DATASET
# ============================================================

def build_samples():

    samples = []

    for class_name in CLASS_NAMES:

        class_dir = IMAGE_ROOT / class_name

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Missing dataset directory: {class_dir}"
            )

        files = sorted(
            class_dir.glob("*.tif")
        )

        print(
            f"{class_name}: "
            f"{len(files)} images"
        )

        for image_path in files:

            samples.append(
                (
                    image_path,
                    CLASS_TO_INDEX[class_name]
                )
            )

    return samples


# ============================================================
# DATASET
# ============================================================

class SARClassificationDataset(
    Dataset
):

    def __init__(
        self,
        samples
    ):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        image = load_sar_image(
            image_path
        )

        image = torch.from_numpy(
            image
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# MODEL
# ============================================================

class SARClassifier(
    nn.Module
):

    def __init__(
        self,
        num_classes=3
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# DATA
# ============================================================

samples = build_samples()

print()

print(
    "Total samples:",
    len(samples)
)

labels = [
    label
    for _, label in samples
]

train_samples, val_samples = (
    train_test_split(
        samples,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)

print(
    "Training samples:",
    len(train_samples)
)

print(
    "Validation samples:",
    len(val_samples)
)


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = (
    SARClassificationDataset(
        train_samples
    )
)

val_dataset = (
    SARClassificationDataset(
        val_samples
    )
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = SARClassifier(
    num_classes=3
)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

MODEL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0

    for images, labels_batch in (
        train_loader
    ):

        images = images.to(
            DEVICE
        )

        labels_batch = labels_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        train_correct += (
            predictions ==
            labels_batch
        ).sum().item()

        train_total += (
            labels_batch.size(0)
        )

    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0

    val_total = 0

    val_loss = 0.0

    with torch.no_grad():

        for images, labels_batch in (
            val_loader
        ):

            images = images.to(
                DEVICE
            )

            labels_batch = (
                labels_batch.to(
                    DEVICE
                )
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels_batch
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            val_correct += (
                predictions ==
                labels_batch
            ).sum().item()

            val_total += (
                labels_batch.size(0)
            )

    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    )


    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT
        )

        print(
            "  Saved best classifier."
        )


print()

print(
    "Training complete."
)

print(
    "Best validation accuracy:",
    round(
        best_val_accuracy,
        4
    )
)

print(
    "Model:",
    MODEL_OUTPUT
)