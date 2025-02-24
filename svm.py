import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Simplified Linear SVM implementation
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)  # Convert to -1, 1

        # Gradient descent
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)


# Main function
def main():
    # Load dataset (use only two classes for simplicity)
    data = load_iris()
    X, y = data.data[data.target != 2, :2], data.target[data.target != 2]  # First two features, binary

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
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
    plt.title('SVM: Iris Classification')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()