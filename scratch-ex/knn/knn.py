"""
k-Nearest Neighbors from scratch.
순수 numpy로 구현한 거리 기반 분류.
"""

import numpy as np
from collections import Counter

np.random.seed(42)

# --- 데이터 생성 ---
# 3개 클래스, 2D 좌표
n_per_class = 50
centers = [(-2, 0), (2, 0), (0, 2)]
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for i, (cx, cy) in enumerate(centers):
    data = np.random.randn(n_per_class, 2) * 0.8 + np.array([cx, cy])
    X_train_list.append(data[:40])
    X_test_list.append(data[40:])
    y_train_list.extend([i] * 40)
    y_test_list.extend([i] * 10)

X_train = np.vstack(X_train_list)  # (120, 2)
X_test = np.vstack(X_test_list)    # (30, 2)
y_train = np.array(y_train_list)
y_test = np.array(y_test_list)

print(f"train: {len(X_train)} samples, test: {len(X_test)} samples")
print(f"classes: {len(centers)}")

# --- kNN ---
def euclidean_distance(a, b):
    """a: (d,), b: (n, d) -> (n,)"""
    return np.sqrt(np.sum((b - a) ** 2, axis=1))

def knn_predict(X_train, y_train, x_query, k):
    distances = euclidean_distance(x_query, X_train)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    # 다수결 투표
    vote = Counter(nearest_labels)
    return vote.most_common(1)[0][0]

def knn_evaluate(X_train, y_train, X_test, y_test, k):
    predictions = [knn_predict(X_train, y_train, x, k) for x in X_test]
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test) * 100
    return accuracy, predictions

# --- 다양한 k 값으로 평가 ---
print("\n--- k 값에 따른 정확도 ---")
for k in [1, 3, 5, 7, 9, 11]:
    acc, _ = knn_evaluate(X_train, y_train, X_test, y_test, k)
    print(f"k={k:2d} | accuracy {acc:.1f}%")

# --- 최적 k로 상세 결과 ---
k_best = 5
acc, preds = knn_evaluate(X_train, y_train, X_test, y_test, k_best)
print(f"\n--- k={k_best} 상세 결과 ---")
for i in range(len(centers)):
    class_mask = y_test == i
    class_acc = np.mean(preds[class_mask] == i) * 100
    print(f"class {i}: {class_acc:.1f}%")

# --- 단일 예측 예시 ---
query = np.array([0.5, 1.5])
pred = knn_predict(X_train, y_train, query, k_best)
distances = euclidean_distance(query, X_train)
nearest_5 = np.argsort(distances)[:k_best]
print(f"\nquery point: {query}")
print(f"prediction: class {pred}")
print(f"nearest {k_best} distances: {distances[nearest_5].round(2)}")
print(f"nearest {k_best} labels: {y_train[nearest_5]}")
