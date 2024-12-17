import os
import numpy as np
import matplotlib.pyplot as plt
from pca_svd import compute_pca_svd

def save_figure(fig, filename):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    fig.savefig(os.path.join('figures', filename))

def reconstruct_faces(data_centered, data_mean, num_components_list, original_images, labels, num_subjects=4):
    unique_labels = np.unique(labels)
    selected_subjects = np.random.choice(unique_labels, num_subjects, replace=False)
    
    fig, axes = plt.subplots(num_subjects, len(num_components_list) + 1, figsize=(15, 5 * num_subjects))
    
    for i, subject in enumerate(selected_subjects):
        subject_indices = np.where(labels == subject)[0]
        subject_image = original_images[subject_indices[0]]
        
        axes[i, 0].imshow(subject_image.reshape(112, 92), cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        for j, n in enumerate(num_components_list):
            print(f"Reconstructing subject {subject} with {n} components...")
            _, eigenvectors_n, _, transformed_data_n = compute_pca_svd(data_centered, num_components=n)
            reconstructed = np.dot(transformed_data_n[subject_indices[0], :n], eigenvectors_n.T) + data_mean
            axes[i, j + 1].imshow(reconstructed.reshape(112, 92), cmap="gray")
            axes[i, j + 1].set_title(f"PCs: {n}")
            axes[i, j + 1].axis("off")
    
    plt.tight_layout()
    save_figure(fig, 'reconstructed_faces.png')
    plt.show()

    