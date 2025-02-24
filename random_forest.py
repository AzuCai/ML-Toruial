import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# Decision Tree Node (for Random Forest)
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Random Forest implementation
class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        features = np.random.choice(X.shape[1], self.n_features, replace=False)
        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gini = (len(y[left_idx]) * self._gini(y[left_idx]) +
                        len(y[right_idx]) * self._gini(y[right_idx])) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=np.bincount(y).argmax())
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.bincount(y).argmax())
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        self.n_features = self.n_features or int(np.sqrt(X.shape[1]))
        for _ in range(self.n_trees):
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree = self._build_tree(X[idx], y[idx])
            self.trees.append(tree)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        tree_preds = np.array([[self._predict_one(x, tree) for x in X] for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, tree_preds)


# Normalize data
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


# Main function
def main():
    # Load dataset
    data = load_wine()
    X, y = data.data, data.target

    # Normalize features
    X = normalize(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForest(n_trees=10, max_depth=3)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize first two features
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='True Labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', cmap='viridis', label='Predictions')
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('Random Forest: Wine Classification')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()