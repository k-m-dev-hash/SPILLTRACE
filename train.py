import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import build_samples, OilSpillDataset
from model import UNet

# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = r"C:\Users\Khushi Sharma\Downloads\02_Test_images_and_ground_truth"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.8

SEED = 42

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "spilltrace_unet.pth"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("SPILLTRACE U-NET TRAINING")
print("=" * 60)

print("Device:", device)


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.view(
            probabilities.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            (2 * intersection + self.smooth)
            /
            (
                probabilities.sum(dim=1)
                +
                targets.sum(dim=1)
                +
                self.smooth
            )
        )

        return 1 - dice.mean()


# ============================================================
# Combined Loss
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return bce_loss + dice_loss


# ============================================================
# Build dataset
# ============================================================

samples = build_samples(
    DATASET_ROOT
)

print()
print("Total samples:", len(samples))


# ============================================================
# Stratified split
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


# Shuffle final sets

random.shuffle(train_samples)
random.shuffle(val_samples)


print()
print("Training samples:", len(train_samples))
print("Validation samples:", len(val_samples))


# ============================================================
# Datasets
# ============================================================

train_dataset = OilSpillDataset(
    train_samples,
    image_size=IMAGE_SIZE
)

val_dataset = OilSpillDataset(
    val_samples,
    image_size=IMAGE_SIZE
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = UNet()

model = model.to(device)


# ============================================================
# Loss + Optimizer
# ============================================================

criterion = CombinedLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


best_val_loss = float("inf")


for epoch in range(EPOCHS):

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 40)


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for batch_index, batch in enumerate(
        train_loader
    ):

        images, masks, categories = batch

        images = images.to(device)
        masks = masks.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            masks
        )


        loss.backward()

        optimizer.step()


        train_loss += loss.item()


        if (
            batch_index + 1
        ) % 20 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )


    train_loss /= len(
        train_loader
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            images, masks, categories = batch

            images = images.to(device)
            masks = masks.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                masks
            )


            val_loss += loss.item()


    val_loss /= len(
        val_loader
    )


    print()
    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best model saved:"
        )

        print(
            MODEL_PATH
        )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Model:",
    MODEL_PATH
)