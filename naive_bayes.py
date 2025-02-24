import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Gaussian Naive Bayes implementation
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) for c in self.classes])
        self.priors = np.array([np.mean(y == c) for c in self.classes])

    def _gaussian(self, X, mean, var):
        return np.exp(-((X - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

    def predict(self, X):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self._gaussian(X, self.mean[i], self.var[i])), axis=1)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors, axis=0)]


# Main function
def main():
    # Load simulated dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = NaiveBayes()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize first two features
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', label='True Labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', cmap='bwr', label='Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Naive Bayes: Spam Classification')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()