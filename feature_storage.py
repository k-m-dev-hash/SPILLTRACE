import numpy as np
import os


def save_features(features, output_path):
    """
    Save processed features as a NumPy file.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, features)

    print(f"Features saved to: {output_path}")


def load_features(input_path):
    """
    Load previously saved features.
    """

    features = np.load(input_path)

    print(f"Features loaded from: {input_path}")
    print(f"Feature shape: {features.shape}")

    return features