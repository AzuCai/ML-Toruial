import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# KNN implementation
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Compute Euclidean distances
        distances = np.sqrt(((self.X_train - X[:, np.newaxis]) ** 2).sum(axis=2))
        # Get indices of k nearest neighbors
        nearest = np.argsort(distances, axis=1)[:, :self.k]
        # Majority vote
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, self.y_train[nearest])


# Main function
def main():
    # Load dataset
    data = load_digits()
    X, y = data.data, data.target

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = KNN(k=3)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize a few test samples
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        ax.set_title(f"Pred: {y_pred[i]}")
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    main()