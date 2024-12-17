import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from load_data import load_images_from_folder
from reconstruct_faces import reconstruct_faces
from additional_methods import lda_method, ica_method, tsne_method, knn_method, logistic_regression_method
from pca_svd import compute_pca_svd
from linear_regression_custom import LinearRegressionCustom

def save_figure(fig, filename):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    fig.savefig(os.path.join('figures', filename))

def main():
    data_folder = '../Data/att_faces'
    images, labels = load_images_from_folder(data_folder)
    data = images.reshape(images.shape[0], -1)
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean

    # PCA using SVD
    num_components = min(data.shape[0], data.shape[1])  # Use the smaller dimension
    eigenvalues, eigenvectors, explained_variance_ratio, transformed_data = compute_pca_svd(data_centered, num_components=num_components)

    # Save eigenvalues
    np.save('eigenvalues.npy', eigenvalues)

    # Show the first 10 leading eigenfaces
    fig = plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(eigenvectors[:, i].reshape(112, 92), cmap="gray")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    plt.tight_layout()
    save_figure(fig, 'leading_eigenfaces.png')
    plt.show()

    # Plot cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.xlabel('Eigen Faces')   
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Eigen Faces')
    plt.grid()
    save_figure(fig, 'cumulative_explained_variance.png')
    plt.show()

    # Plot percentage of variance
    fig = plt.figure(figsize=(10, 5))
    plt.plot(explained_variance_ratio * 100, marker='o')
    plt.xlabel('Eigen Faces')   
    plt.ylabel('Percentage of Variance')
    plt.title('Percentage of Variance vs. Eigen Faces')
    plt.grid()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    save_figure(fig, 'percentage_of_variance.png')
    plt.show()

    # Reconstruct faces
    num_components_list = [10, 50, 150, 250, 350, 400]
    reconstruct_faces(data_centered, data_mean, num_components_list, images, labels, num_subjects=4)

    # Additional methods
    print("Applying LDA...")
    train_lda, test_lda = lda_method(data_centered, labels, data_centered, n_components=30)
    print("Applying ICA...")
    train_ica, test_ica = ica_method(data_centered, data_centered, n_components=30)
    print("Applying t-SNE...")
    train_tsne, test_tsne = tsne_method(data_centered, data_centered, n_components=2)
    
    # Feature scaling
    scaler = StandardScaler()
    train_lda = scaler.fit_transform(train_lda)
    test_lda = scaler.transform(test_lda)

    # Classification using KNN
    print("Applying KNN...")
    knn_predictions = knn_method(train_lda, labels, test_lda, k=5)
    knn_accuracy = accuracy_score(labels, knn_predictions)
    print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

    # Classification using Logistic Regression
    print("Applying Logistic Regression...")
    log_reg_predictions = logistic_regression_method(train_lda, labels, test_lda)
    log_reg_accuracy = accuracy_score(labels, log_reg_predictions)
    print(f"Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%")

    # Classification using Custom Linear Regression
    print("Applying Custom Linear Regression...")
    custom_lr = LinearRegressionCustom()
    custom_lr.fit(train_lda, labels)
    custom_lr_predictions = custom_lr.predict(test_lda)
    custom_lr_predictions_rounded = np.round(custom_lr_predictions).astype(int)
    custom_lr_accuracy = accuracy_score(labels, custom_lr_predictions_rounded)
    print(f"Custom Linear Regression Accuracy: {custom_lr_accuracy * 100:.2f}%")

    # Calculate MSE for PCA reconstruction
    mse_list = []
    for n in num_components_list:
        _, eigenvectors_n, _, transformed_data_n = compute_pca_svd(data_centered, num_components=n)
        reconstructed = np.dot(transformed_data_n, eigenvectors_n.T) + data_mean
        mse = mean_squared_error(data, reconstructed)
        mse_list.append(mse)
        print(f"MSE for {n} components: {mse:.4f}")

    # Display results of additional methods
    print("Applying additional methods and displaying results...")
    methods = {
        "KNN": knn_accuracy,
        "Logistic Regression": log_reg_accuracy,
        "Custom Linear Regression": custom_lr_accuracy
    }

    for method, accuracy in methods.items():
        print(f"{method} Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

