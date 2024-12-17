import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_images_from_folder
from linear_regression_custom import LinearRegressionCustom
from additional_methods import lda_method, kpca_method
from pca_svd import compute_pca_svd

def classify_faces(data_folder):
    images, labels = load_images_from_folder(data_folder)
    print(f"Loaded {len(images)} images.")

    # Reshape images to 2D array
    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)

    # Split into train and test sets
    split_ratio = 0.8
    split_index = int(n_samples * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    # Apply PCA using SVD
    num_components = min(X_train.shape[0], X_train.shape[1]) 
    eigenvalues, eigenvectors, explained_variance_ratio, X_train_pca = compute_pca_svd(X_train, num_components=num_components)
    X_test_pca = np.dot(X_test, eigenvectors)

    classifiers = {
        "Custom Linear Regression": LinearRegressionCustom(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f'{name} Accuracy: {accuracy * 100:.2f}%')

    # Apply additional methods
    print("Applying LDA...")
    X_train_lda, X_test_lda = lda_method(X_train, y_train, X_test, n_components=30)
    print("Applying KPCA...")
    X_train_kpca, X_test_kpca = kpca_method(X_train, X_test, n_components=200, kernel="rbf")

    # Classification using LDA
    print("Classifying using LDA...")
    for name, clf in classifiers.items():
        print(f"Training {name} with LDA...")
        clf.fit(X_train_lda, y_train)
        y_pred = clf.predict(X_test_lda)
        accuracy = accuracy_score(y_test, y_pred)
        results[f"{name} with LDA"] = accuracy
        print(f'{name} with LDA Accuracy: {accuracy * 100:.2f}%')

    # Classification using KPCA
    print("Classifying using KPCA...")
    for name, clf in classifiers.items():
        print(f"Training {name} with KPCA...")
        clf.fit(X_train_kpca, y_train)
        y_pred = clf.predict(X_test_kpca)
        accuracy = accuracy_score(y_test, y_pred)
        results[f"{name} with KPCA"] = accuracy
        print(f'{name} with KPCA Accuracy: {accuracy * 100:.2f}%')

    # Save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/classification_results.txt', 'w') as f:
        for name, accuracy in results.items():
            f.write(f'{name} Accuracy: {accuracy * 100:.2f}%\n')
        f.write(classification_report(y_test, y_pred, target_names=[str(i) for i in np.unique(labels)]))

    # Display results of all methods
    print("\nSummary of Classification Results:")
    for name, accuracy in results.items():
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    data_folder = '../Data/att_faces'
    classify_faces(data_folder)

    import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_images_from_folder
from linear_regression_custom import LinearRegressionCustom
from additional_methods import lda_method, kpca_method
from pca_svd import compute_pca_svd

def classify_faces(data_folder):
    images, labels = load_images_from_folder(data_folder)
    print(f"Loaded {len(images)} images.")

    # Reshape images to 2D array
    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)

    # Split into train and test sets
    split_ratio = 0.8
    split_index = int(n_samples * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    # Apply PCA using SVD
    num_components = min(X_train.shape[0], X_train.shape[1]) 
    eigenvalues, eigenvectors, explained_variance_ratio, X_train_pca = compute_pca_svd(X_train, num_components=num_components)
    X_test_pca = np.dot(X_test, eigenvectors)

    classifiers = {
        "Custom Linear Regression": LinearRegressionCustom(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f'{name} Accuracy: {accuracy * 100:.2f}%')

    # Apply additional methods
    print("Applying LDA...")
    X_train_lda, X_test_lda = lda_method(X_train, y_train, X_test, n_components=30)
    print("Applying KPCA...")
    X_train_kpca, X_test_kpca = kpca_method(X_train, X_test, n_components=200, kernel="rbf")

    # Classification using LDA
    print("Classifying using LDA...")
    for name, clf in classifiers.items():
        print(f"Training {name} with LDA...")
        clf.fit(X_train_lda, y_train)
        y_pred = clf.predict(X_test_lda)
        accuracy = accuracy_score(y_test, y_pred)
        results[f"{name} with LDA"] = accuracy
        print(f'{name} with LDA Accuracy: {accuracy * 100:.2f}%')

    # Classification using KPCA
    print("Classifying using KPCA...")
    for name, clf in classifiers.items():
        print(f"Training {name} with KPCA...")
        clf.fit(X_train_kpca, y_train)
        y_pred = clf.predict(X_test_kpca)
        accuracy = accuracy_score(y_test, y_pred)
        results[f"{name} with KPCA"] = accuracy
        print(f'{name} with KPCA Accuracy: {accuracy * 100:.2f}%')

    # Save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/classification_results.txt', 'w') as f:
        for name, accuracy in results.items():
            f.write(f'{name} Accuracy: {accuracy * 100:.2f}%\n')
        f.write(classification_report(y_test, y_pred, target_names=[str(i) for i in np.unique(labels)]))

    # Display results of all methods
    print("\nSummary of Classification Results:")
    for name, accuracy in results.items():
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    data_folder = '../Data/att_faces'
    classify_faces(data_folder)

