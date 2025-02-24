import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine


# K-Means implementation
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # Randomly initialize centroids
        idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # Assign clusters
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
        return labels


# Main function
def main():
    # Load dataset
    data = load_wine()
    X = data.data[:, :2]  # Use first two features for visualization

    # Train model
    model = KMeans(k=3)
    labels = model.fit(X)

    # Visualize
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Clusters')
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='x', label='Centroids')
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('K-Means: Wine Clustering')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()