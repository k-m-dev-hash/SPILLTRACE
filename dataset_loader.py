import os
import numpy as np
import tifffile
import cv2


def load_sentinel_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = tifffile.imread(image_path)

    print("Image path:", image_path)
    print("Image shape:", image.shape)
    print("Data type:", image.dtype)

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D Sentinel-1 image, got shape {image.shape}"
        )

    # Sentinel-1 dataset: VV and VH are the two channels
    if image.shape[-1] == 2:
        vv = image[:, :, 0]
        vh = image[:, :, 1]

    elif image.shape[0] == 2:
        vv = image[0]
        vh = image[1]

    else:
        raise ValueError(
            f"Could not identify VV/VH channels from shape {image.shape}"
        )

    return {
        "image": image,
        "vv": vv,
        "vh": vh,
        "height": vv.shape[0],
        "width": vv.shape[1]
    }


def get_image_statistics(vv, vh):
    return {
        "vv_min": float(np.min(vv)),
        "vv_max": float(np.max(vv)),
        "vv_mean": float(np.mean(vv)),
        "vv_std": float(np.std(vv)),
        "vh_min": float(np.min(vh)),
        "vh_max": float(np.max(vh)),
        "vh_mean": float(np.mean(vh)),
        "vh_std": float(np.std(vh))
    }
def save_vv_preview(vv, output_path):
    # Convert VV values to 0–255 for visualization
    normalized = cv2.normalize(
        vv,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    normalized = normalized.astype(np.uint8)

    cv2.imwrite(output_path, normalized)

    print("Preview saved to:", output_path)