import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child
        self.right = right  # Right child
        self.value = value  # Value for leaf node

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def entropy(self, target):
        n_samples = len(target)
        unique, counts = np.unique(target, return_counts=True)
        probabilities = counts / n_samples
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def weighted_entropy(self, feature, target):
        unique_values = np.unique(feature)
        weighted_entropy = 0
        for value in unique_values:
            subset_index = np.where(feature == value)[0]
            subset_target = target[subset_index]
            subset_entropy = self.entropy(subset_target)
            weight = len(subset_target) / len(target)
            weighted_entropy += weight * subset_entropy
        return weighted_entropy

    def information_gain(self, feature, target):
        parent_entropy = self.entropy(target)
        weighted_entropy = self.weighted_entropy(feature, target)
        return parent_entropy - weighted_entropy

    def best_split(self, dataset, features, target):
        max_gain = -1
        best_feature = None
        for feature in features:
            feature_column = dataset[:, feature]
            gain = self.information_gain(feature_column, target)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature

    def build_tree(self, dataset, features, target, depth=0):
        if len(np.unique(target)) == 1:  # Pure leaf
            return Node(value=target[0])
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            most_common_value = np.bincount(target).argmax()
            return Node(value=most_common_value)

        best_feature = self.best_split(dataset, features, target)
        if best_feature is None:
            most_common_value = np.bincount(target).argmax()
            return Node(value=most_common_value)

        feature_column = dataset[:, best_feature]
        unique_values = np.unique(feature_column)
        threshold = np.median(unique_values)

        left_index = feature_column <= threshold
        right_index = feature_column > threshold

        left = self.build_tree(dataset[left_index], features, target[left_index], depth + 1)
        right = self.build_tree(dataset[right_index], features, target[right_index], depth + 1)

        return Node(feature=best_feature, threshold=threshold, left=left, right=right)

    def fit(self, dataset, target):
        features = np.arange(dataset.shape[1])
        self.root = self.build_tree(dataset, features, target)

    def predict_sample(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self.predict_sample(node.left, sample)
        else:
            return self.predict_sample(node.right, sample)

    def predict(self, dataset):
        return np.array([self.predict_sample(self.root, sample) for sample in dataset])

def main():
    # Example Dataset
    X = np.array([[2.7, 2.5], [1.3, 12.5], [10.6, 4.2], [4.1, 4.7], [9.0, 1.9]])
    y = np.array([0, 0, 1, 1, 0])

    # Create Decision Tree
    tree = DecisionTree(max_depth=3)

    # Fit the tree on the dataset
    print("Training the Decision Tree...")
    tree.fit(X, y)

    # Predict on the training dataset
    predictions = tree.predict(X)
    print("\nPredictions on the training dataset:")
    print(predictions)

    # Evaluate accuracy
    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy on training dataset: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
