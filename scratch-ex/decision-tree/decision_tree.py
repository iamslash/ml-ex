"""
Decision Tree from scratch.
순수 numpy로 구현한 결정 트리 (분류).
"""

import numpy as np
from collections import Counter

np.random.seed(42)

# --- 데이터 생성 ---
# Iris-like 합성 데이터: 3클래스, 4특성
n_per_class = 50
n_classes = 3

# X0, X1, X2: (n_per_class, 4) — 클래스별 특성 행렬
X0 = np.random.randn(n_per_class, 4) * 0.5 + np.array([1, 1, 0, 0])
X1 = np.random.randn(n_per_class, 4) * 0.5 + np.array([0, 0, 2, 2])
X2 = np.random.randn(n_per_class, 4) * 0.5 + np.array([2, 0, 1, 3])

# X: (n_per_class * n_classes, 4), y: (n_per_class * n_classes,)
X = np.vstack([X0, X1, X2])
y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]  # 모양 동일

# train/test split — X_train: (split, 4), X_test: (N-split, 4), y_*: (split,) / (N-split,)
split = 120
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 예상 출력: train: 120, test: 30, features: 4, classes: 3
print(f"train: {len(X_train)}, test: {len(X_test)}, features: {X.shape[1]}, classes: {n_classes}")

# --- 엔트로피 ---
# y: (n_samples,) — 정수 클래스 라벨
def entropy(y):
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

# --- 정보 이득 ---
# y, left_mask: (n_samples,) — left_mask는 bool
def information_gain(y, left_mask):
    if left_mask.sum() == 0 or (~left_mask).sum() == 0:
        return 0
    parent_entropy = entropy(y)
    n = len(y)
    n_left = left_mask.sum()
    n_right = n - n_left
    child_entropy = (n_left / n) * entropy(y[left_mask]) + (n_right / n) * entropy(y[~left_mask])
    return parent_entropy - child_entropy

# --- 최적 분할 찾기 ---
# X: (n_samples, n_features), y: (n_samples,)
def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            gain = information_gain(y, left_mask)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

# --- 트리 노드 ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # 분할 기준 특성 인덱스
        self.threshold = threshold    # 분할 기준 값
        self.left = left              # 왼쪽 자식 (<=)
        self.right = right            # 오른쪽 자식 (>)
        self.value = value            # 리프 노드의 예측 클래스

# --- 트리 구축 ---
# X: (n_samples, n_features), y: (n_samples,)
def build_tree(X, y, depth=0, max_depth=5):
    # 종료 조건: 순수 노드 또는 최대 깊이
    if len(np.unique(y)) == 1:
        return Node(value=y[0])
    if depth >= max_depth or len(y) < 2:
        return Node(value=Counter(y).most_common(1)[0][0])

    feature, threshold, gain = find_best_split(X, y)

    if gain == 0:
        return Node(value=Counter(y).most_common(1)[0][0])

    left_mask = X[:, feature] <= threshold
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left, right=right)

# --- 예측 ---
# x: (n_features,) — 단일 샘플 특성 벡터
def predict_one(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

# X: (n_samples, n_features) → 반환 (n_samples,)
def predict(node, X):
    return np.array([predict_one(node, x) for x in X])

# --- 트리 출력 ---
def print_tree(node, indent=""):
    if node.value is not None:
        print(f"{indent}-> class {node.value}")
        return
    print(f"{indent}feature[{node.feature}] <= {node.threshold:.2f}?")
    print(f"{indent}  yes:")
    print_tree(node.left, indent + "    ")
    print(f"{indent}  no:")
    print_tree(node.right, indent + "    ")

# --- 학습 ---
print("\nbuilding tree...")
tree = build_tree(X_train, y_train, max_depth=4)

print("\n--- tree structure ---")
print_tree(tree)

# --- 평가 ---
# train_preds: (len(X_train),), test_preds: (len(X_test),)
train_preds = predict(tree, X_train)
test_preds = predict(tree, X_test)

train_acc = np.mean(train_preds == y_train) * 100
test_acc = np.mean(test_preds == y_test) * 100

print(f"\ntrain accuracy: {train_acc:.1f}%")
print(f"test  accuracy: {test_acc:.1f}%")

for i in range(n_classes):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = np.mean(test_preds[mask] == i) * 100
        print(f"  class {i}: {class_acc:.1f}%")
