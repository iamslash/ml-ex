"""
Logistic Regression from scratch.
순수 numpy로 구현한 이진 분류.
"""

import numpy as np

np.random.seed(42)

# --- 데이터 생성 ---
# 2D 점들을 두 그룹으로 분류
n_samples = 200
n_half = n_samples // 2

# 그룹 0: 중심 (-1, -1)
X0 = np.random.randn(n_half, 2) * 0.8 + np.array([-1, -1])
# 그룹 1: 중심 (1, 1)
X1 = np.random.randn(n_half, 2) * 0.8 + np.array([1, 1])

X = np.vstack([X0, X1])        # (200, 2)
y = np.array([0]*n_half + [1]*n_half).reshape(-1, 1)  # (200, 1)

# shuffle
indices = np.random.permutation(n_samples)
X, y = X[indices], y[indices]

print(f"data: {n_samples} samples, {X.shape[1]} features")
print(f"class 0: {(y==0).sum()}, class 1: {(y==1).sum()}")

# --- sigmoid ---
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# --- 모델 파라미터 초기화 ---
# P(y=1|x) = sigmoid(x @ w + b)
w = np.zeros((2, 1))  # (2, 1)
b = np.zeros((1, 1))  # (1, 1)

# --- 하이퍼파라미터 ---
learning_rate = 0.5
num_epochs = 200

# --- 학습 ---
for epoch in range(num_epochs):
    # Forward: 예측 확률
    z = X @ w + b           # (200, 1)
    y_pred = sigmoid(z)     # (200, 1)

    # Loss: Binary Cross-Entropy
    # L = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
    eps = 1e-7
    loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

    # Backward: 그래디언트
    # dL/dw = (1/n) * X^T @ (y_pred - y)
    # dL/db = (1/n) * sum(y_pred - y)
    error = y_pred - y  # (200, 1)
    dw = (1 / n_samples) * X.T @ error  # (2, 1)
    db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)  # (1, 1)

    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 20 == 0:
        acc = np.mean((y_pred >= 0.5).astype(int) == y) * 100
        print(f"epoch {epoch:3d} | loss {loss:.4f} | accuracy {acc:.1f}%")

# --- 최종 결과 ---
y_pred = sigmoid(X @ w + b)
acc = np.mean((y_pred >= 0.5).astype(int) == y) * 100
print(f"\nfinal accuracy: {acc:.1f}%")
print(f"w = [{w[0,0]:.4f}, {w[1,0]:.4f}], b = {b[0,0]:.4f}")
