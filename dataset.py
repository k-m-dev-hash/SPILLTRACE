import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):

    def __init__(self, samples, image_size=256):
        """
        samples:
            List of tuples:
            (image_path, mask_path, category)

        category:
            Oil
            Lookalike
            No oil
        """

        self.samples = samples
        self.image_size = image_size

        if len(self.samples) == 0:
            raise ValueError("No samples provided.")

        print(f"Dataset samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path, category = self.samples[index]

        # --------------------------------------------------
        # Load Sentinel-1 image
        # --------------------------------------------------

        image = tifffile.imread(image_path)

        if image.ndim != 3:
            raise ValueError(
                f"Unexpected image shape: {image.shape}"
            )

        # Expected:
        # (2048, 2048, 2)

        if image.shape[-1] == 2:

            vv = image[:, :, 0]
            vh = image[:, :, 1]

        elif image.shape[0] == 2:

            vv = image[0]
            vh = image[1]

        else:

            raise ValueError(
                f"Cannot identify VV/VH channels: {image.shape}"
            )

        # --------------------------------------------------
        # Replace invalid values
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Normalize VV and VH
        # --------------------------------------------------

        vv_mean = np.mean(vv)
        vv_std = np.std(vv)

        vh_mean = np.mean(vh)
        vh_std = np.std(vh)

        vv = (vv - vv_mean) / (vv_std + 1e-6)
        vh = (vh - vh_mean) / (vh_std + 1e-6)

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        import cv2

        vv = cv2.resize(
            vv,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        vh = cv2.resize(
            vh,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        # --------------------------------------------------
        # Stack VV + VH
        # --------------------------------------------------

        image = np.stack(
            [vv, vh],
            axis=0
        ).astype(np.float32)

        # --------------------------------------------------
        # Load mask
        # --------------------------------------------------

        mask = tifffile.imread(mask_path)

        mask = (mask > 0).astype(np.float32)

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
            category
        )


def build_samples(dataset_root):

    images_root = os.path.join(
        dataset_root,
        "Images"
    )

    masks_root = os.path.join(
        dataset_root,
        "Mask"
    )

    categories = [
        "Oil",
        "Lookalike",
        "No oil"
    ]

    samples = []

    for category in categories:

        image_dir = os.path.join(
            images_root,
            category
        )

        mask_dir = os.path.join(
            masks_root,
            category
        )

        if not os.path.exists(image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        if not os.path.exists(mask_dir):
            raise FileNotFoundError(
                f"Mask directory not found: {mask_dir}"
            )

        image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith(".tif")
            ]
        )

        for image_name in image_files:

            base_name = os.path.splitext(
                image_name
            )[0]

            mask_name = (
                base_name
                + "_segmentation.tif"
            )

            image_path = os.path.join(
                image_dir,
                image_name
            )

            mask_path = os.path.join(
                mask_dir,
                mask_name
            )

            if not os.path.exists(mask_path):
                print(
                    f"Warning: mask missing for {image_name}"
                )
                continue

            samples.append(
                (
                    image_path,
                    mask_path,
                    category
                )
            )

    return samples