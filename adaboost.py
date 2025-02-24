import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Decision Stump (weak learner)
class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, feature] < polarity * threshold] = -1
                    error = np.sum(weights * (pred != y))
                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
        self.alpha = 0.5 * np.log((1 - min_error) / (max(min_error, 1e-10)))

    def predict(self, X):
        pred = np.ones(X.shape[0])
        pred[self.polarity * X[:, self.feature] < self.polarity * self.threshold] = -1
        return pred


# AdaBoost implementation
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples
        y_ = np.where(y <= 0, -1, 1)  # Convert to -1, 1

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y_, weights)
            pred = stump.predict(X)
            weights *= np.exp(-stump.alpha * y_ * pred)
            weights /= weights.sum()
            self.models.append(stump)
            self.alphas.append(stump.alpha)

    def predict(self, X):
        pred = sum(alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models))
        return np.sign(pred)


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
    model = AdaBoost(n_estimators=50)
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
    plt.title('AdaBoost: Breast Cancer Classification')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()