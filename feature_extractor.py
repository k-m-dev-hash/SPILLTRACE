import numpy as np


def extract_features(vv, vh):
    """
    Extract pixel-level features from Sentinel-1 VV and VH bands.

    Features:
    1. VV
    2. VH
    3. VV/VH ratio
    4. VV-VH difference
    """

    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)

    # Avoid division by zero
    ratio = vv / (vh + 1e-6)

    difference = vv - vh

    features = np.stack(
        [
            vv,
            vh,
            ratio,
            difference
        ],
        axis=-1
    )

    return features


def get_feature_statistics(features):
    """
    Display basic statistics of extracted features.
    """

    return {
        "shape": features.shape,
        "mean": np.mean(features, axis=(0, 1)),
        "std": np.std(features, axis=(0, 1)),
        "min": np.min(features, axis=(0, 1)),
        "max": np.max(features, axis=(0, 1))
    }