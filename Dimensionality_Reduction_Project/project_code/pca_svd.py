import numpy as np

def compute_pca_svd(data, num_components=None):
    # Center the data
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean

    # Compute SVD
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

    # Select the top num_components
    if num_components:
        U = U[:, :num_components]
        S = S[:num_components]
        Vt = Vt[:num_components, :]

    # Compute eigenvalues and eigenvectors
    eigenvalues = S**2 / (data_centered.shape[0] - 1)
    eigenvectors = Vt.T

    # Project the data onto the selected eigenvectors
    transformed_data = np.dot(data_centered, eigenvectors)

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    return eigenvalues, eigenvectors, explained_variance_ratio, transformed_data
