import torch
import numpy as np

# Generate synthetic XOR data using four different Gaussians
def generate_xor_data(n_samples):
    n_samples_per_quadrant = n_samples // 4
    var = 16
    offset = 64
    clip = 127
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, offset],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, -offset],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, offset],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, -offset]  # Bottom-right
    ])
    X = np.clip(X, -clip, clip)

    y = np.hstack([
        -np.ones(n_samples_per_quadrant),  # Top-right
        -np.ones(n_samples_per_quadrant),  # Bottom-left
        np.ones(n_samples_per_quadrant),  # Top-left
        np.ones(n_samples_per_quadrant)  # Bottom-right
    ])

    return torch.tensor(X, dtype=torch.int8), torch.tensor(y, dtype=torch.float32)

# Generate synthetic linear data using four different Gaussians
def generate_linear_data(n_samples):
    n_samples_per_quadrant = n_samples // 4
    var = 16
    offset = 64
    clip = 127
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, offset],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, -offset],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-offset, offset],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [offset, -offset]  # Bottom-right
    ])
    X = np.clip(X, -clip, clip)

    y = np.hstack([
        -np.ones(n_samples_per_quadrant),  # Top-right
        np.ones(n_samples_per_quadrant),  # Bottom-left
        np.ones(n_samples_per_quadrant),  # Top-left
        -np.ones(n_samples_per_quadrant)  # Bottom-right
    ])
    return torch.tensor(X, dtype=torch.int8), torch.tensor(y, dtype=torch.float32)
