import numpy as np

class LinearRegressionCustom:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        # Compute weights using the normal equation
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Add bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights
    
    