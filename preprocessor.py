import numpy as np


def clean_features(features):
    """
    Clean Sentinel-1 features and remove invalid/extreme values.
    """

    features = np.asarray(features, dtype=np.float32).copy()

    # Replace NaN and infinite values
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Clip each feature using percentile-based limits
    for i in range(features.shape[-1]):
        low = np.percentile(features[:, :, i], 1)
        high = np.percentile(features[:, :, i], 99)

        features[:, :, i] = np.clip(
            features[:, :, i],
            low,
            high
        )

    return features


def normalize_features(features):
    """
    Normalize each feature to approximately 0-1.
    """

    features = np.asarray(features, dtype=np.float32)

    minimum = np.min(features, axis=(0, 1), keepdims=True)
    maximum = np.max(features, axis=(0, 1), keepdims=True)

    normalized = (features - minimum) / (
        maximum - minimum + 1e-8
    )

    return normalized


def preprocess_features(features):
    """
    Complete preprocessing pipeline.
    """

    cleaned = clean_features(features)
    normalized = normalize_features(cleaned)

    return normalized