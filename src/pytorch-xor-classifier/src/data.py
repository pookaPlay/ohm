import torch
import numpy as np

# Generate synthetic XOR data using four different Gaussians
def generate_xor_data(n_samples):
    n_samples_per_quadrant = n_samples // 4
    var = 0.5
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [2, 2],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-2, -2],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-2, 2],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [2, -2]  # Bottom-right
    ])
    y = np.hstack([
        np.zeros(n_samples_per_quadrant),  # Top-right
        np.zeros(n_samples_per_quadrant),  # Bottom-left
        np.ones(n_samples_per_quadrant),  # Top-left
        np.ones(n_samples_per_quadrant)  # Bottom-right
    ])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def generate_linear_data(n_samples):
    n_samples_per_quadrant = n_samples // 4
    var = 0.5
    X = np.vstack([
        var * np.random.randn(n_samples_per_quadrant, 2) + [2, 2],  # Top-right
        var * np.random.randn(n_samples_per_quadrant, 2) + [-2, -2],  # Bottom-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [-2, 2],  # Top-left
        var * np.random.randn(n_samples_per_quadrant, 2) + [2, -2]  # Bottom-right
    ])
    y = np.hstack([
        np.zeros(n_samples_per_quadrant),  # Top-right
        np.ones(n_samples_per_quadrant),  # Bottom-left
        np.ones(n_samples_per_quadrant),  # Top-left
        np.zeros(n_samples_per_quadrant)  # Bottom-right
    ])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


