import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value


# Decision Tree implementation
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(X.shape[1]):
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
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# Main function
def main():
    # Load dataset
    titanic = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'survived'])
    X = titanic[['age', 'fare']].values
    y = titanic['survived'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTree(max_depth=3)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', label='True Labels')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', cmap='bwr', label='Predictions')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.title('Decision Tree: Titanic Survival')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()