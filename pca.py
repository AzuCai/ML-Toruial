import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces


# PCA implementation
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov = np.cov(X_centered.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered.dot(self.components)


# Main function
def main():
    # Load dataset
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data

    # Train model
    model = PCA(n_components=2)
    model.fit(X)
    X_pca = model.transform(X)

    # Visualize
    plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: Face Dataset Dimensionality Reduction')
    plt.show()


if __name__ == "__main__":
    main()