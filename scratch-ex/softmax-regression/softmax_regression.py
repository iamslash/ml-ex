"""
Softmax Regression from scratch.
순수 numpy로 구현한 다중 분류.
"""

import numpy as np

np.random.seed(42)

# --- 데이터 생성 ---
# 3개 클래스, 2D 좌표
n_per_class = 100
n_classes = 3
n_samples = n_per_class * n_classes

centers = [(-2, 0), (2, 0), (0, 2)]
X_list, y_list = [], []

for i, (cx, cy) in enumerate(centers):
    X_list.append(np.random.randn(n_per_class, 2) * 0.6 + np.array([cx, cy]))
    y_list.append(np.full(n_per_class, i))

X = np.vstack(X_list)           # (300, 2)
y = np.concatenate(y_list)       # (300,)

# shuffle
indices = np.random.permutation(n_samples)
X, y = X[indices], y[indices]

# one-hot encoding
y_onehot = np.zeros((n_samples, n_classes))  # (300, 3)
y_onehot[np.arange(n_samples), y] = 1

print(f"data: {n_samples} samples, {X.shape[1]} features, {n_classes} classes")

# --- softmax ---
def softmax(z):
    z_shift = z - z.max(axis=1, keepdims=True)  # 수치 안정성
    exp_z = np.exp(z_shift)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# --- 모델 파라미터 초기화 ---
# P(y=k|x) = softmax(x @ W + b)[k]
W = np.zeros((2, n_classes))     # (2, 3)
b = np.zeros((1, n_classes))     # (1, 3)

# --- 하이퍼파라미터 ---
learning_rate = 0.5
num_epochs = 300

# --- 학습 ---
for epoch in range(num_epochs):
    # Forward
    z = X @ W + b               # (300, 3)
    probs = softmax(z)          # (300, 3)

    # Loss: Cross-Entropy
    # L = -(1/n) * Σ Σ y_k * log(p_k)
    eps = 1e-7
    loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

    # Backward
    # dL/dz = (1/n) * (probs - y_onehot)
    error = (probs - y_onehot) / n_samples  # (300, 3)
    dW = X.T @ error            # (2, 3)
    db = np.sum(error, axis=0, keepdims=True)  # (1, 3)

    # Update
    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 30 == 0:
        predictions = np.argmax(probs, axis=1)
        acc = np.mean(predictions == y) * 100
        print(f"epoch {epoch:3d} | loss {loss:.4f} | accuracy {acc:.1f}%")

# --- 최종 결과 ---
probs = softmax(X @ W + b)
predictions = np.argmax(probs, axis=1)
acc = np.mean(predictions == y) * 100
print(f"\nfinal accuracy: {acc:.1f}%")

for i in range(n_classes):
    class_acc = np.mean(predictions[y == i] == i) * 100
    print(f"  class {i}: {class_acc:.1f}%")
