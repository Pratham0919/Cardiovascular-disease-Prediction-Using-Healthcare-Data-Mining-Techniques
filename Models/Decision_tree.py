import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_and_preprocess_data():

    df = pd.read_csv('Dataset/cardio_clean.csv')
    
    X = df.drop(columns=['cardio'])  
    y = df['cardio']
    return X, y


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        
        if len(unique_classes) == 1:
            return {'label': unique_classes[0]}

        
        if self.max_depth and depth == self.max_depth:
            most_common_class = self._most_common_class(y)
            return {'label': most_common_class}

        
        split_best = self._best_split(X, y)
        lefttree = self._build_tree(split_best['left_X'], split_best['left_y'], depth + 1)
        righttree = self._build_tree(split_best['right_X'], split_best['right_y'], depth + 1)

        return {
            'feature_index': split_best['feature_index'],
            'threshold': split_best['threshold'],
            'left': lefttree,
            'right': righttree
        }

    def _best_split(self, X, y):
        best_gain_info = -float('inf')
        best_split = {}

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            values_of_features = X[:, feature_index]
            thresholds_possibilities = np.unique(values_of_features)

            for threshold in thresholds_possibilities:
                indices_left = values_of_features <= threshold
                indices_right = values_of_features > threshold

                if np.sum(indices_left) == 0 or np.sum(indices_right) == 0:
                    continue

                left_y = y[indices_left]
                right_y = y[indices_right]

                
                gain_info = self._information_gain(y, left_y, right_y)

                if gain_info > best_gain_info:
                    best_gain_info = gain_info
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_X': X[indices_left],
                        'left_y': left_y,
                        'right_X': X[indices_right],
                        'right_y': right_y
                    }

        return best_split
    
    def _information_gain(self, y, left_y, right_y):
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        probabilities = [count / len(y) for count in class_counts.values()]
        return -sum(p * self._log2(p) for p in probabilities)

    def _log2(self, x):
        return 0 if x == 0 else math.log(x, 2)

    def _most_common_class(self, y):
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        return max(class_counts, key=class_counts.get)

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _predict_one(self, x, tree):
        if 'label' in tree:
            return tree['label']

        feature_value = x[tree['feature_index']]
        
        if feature_value <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])


def main():
    
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTree(max_depth=5)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    predictions = model.predict(X_test.to_numpy())
    
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, predictions))
    
    

if __name__ == "__main__":
    main()
