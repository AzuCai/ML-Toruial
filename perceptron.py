import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Perceptron implementation
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1, 1

        # Training loop
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.sign(linear_output)
                if y_[idx] * linear_output <= 0:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)


# Normalize data
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


# Main function
def main():
    # Load real dataset (Breast Cancer)
    data = load_breast_cancer()
    X, y = data.data[:, :2], data.target  # Use first two features for simplicity

    # Normalize features
    X = normalize(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = Perceptron(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == np.where(y_test <= 0, -1, 1))
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', label='True Labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', cmap='bwr', label='Predictions')
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('Perceptron: Breast Cancer Classification')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()