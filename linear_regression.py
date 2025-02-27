import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# Linear Regression implementation
class LinearRegression:
    def __init__(self):
        self.weights = None  # Weights for features
        self.bias = None  # Bias term

    def fit(self, X, y):
        # Add bias term to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Solve normal equation: (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.bias, self.weights])


# Main function
def main():
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

    # Sample 100 points for visualization to reduce clutter
    sample_size = 100
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    y_test_sample = y_test[indices]
    y_pred_sample = y_pred[indices]

    # Visualize predictions vs true values with transparency
    plt.scatter(y_test_sample, y_pred_sample, color='blue', alpha=0.5, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Linear Regression: California Housing Prices (Sampled)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
