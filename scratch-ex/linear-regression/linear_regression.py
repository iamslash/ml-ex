"""
Linear Regression from scratch.
순수 Python + numpy로 구현한 선형 회귀.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 데이터 생성 ---
# y = 3x + 7 + noise
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)           # (100, 1) 0~2 사이 값
y = 3 * X + 7 + np.random.randn(n_samples, 1) * 0.5  # (100, 1)

print(f"데이터: {n_samples}개 샘플")
print(f"X 범위: [{X.min():.2f}, {X.max():.2f}]")
print(f"y 범위: [{y.min():.2f}, {y.max():.2f}]")

# --- 모델 파라미터 초기화 ---
# y = w * x + b
w = np.random.randn(1, 1) * 0.01  # 가중치
b = np.zeros((1, 1))               # 편향

# --- 하이퍼파라미터 ---
learning_rate = 0.1
num_epochs = 100

# --- 학습 ---
losses = []

for epoch in range(num_epochs):
    # Forward: 예측값 계산
    y_pred = X @ w + b  # (100, 1)

    # Loss: MSE (Mean Squared Error)
    error = y_pred - y
    loss = np.mean(error ** 2)
    losses.append(loss)

    # Backward: 그래디언트 계산
    # dL/dw = (2/n) * X^T @ (y_pred - y)
    # dL/db = (2/n) * sum(y_pred - y)
    dw = (2 / n_samples) * X.T @ error  # (1, 1)
    db = (2 / n_samples) * np.sum(error, axis=0, keepdims=True)  # (1, 1)

    # Update: 파라미터 갱신
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d} | loss {loss:.4f} | w={w[0,0]:.4f} b={b[0,0]:.4f}")

print(f"\n최종 파라미터: w={w[0,0]:.4f}, b={b[0,0]:.4f}")
print(f"정답 파라미터: w=3.0000, b=7.0000")

# --- 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 회귀선
axes[0].scatter(X, y, alpha=0.5, label='data')
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
y_line = X_line @ w + b
axes[0].plot(X_line, y_line, 'r-', linewidth=2, label=f'y={w[0,0]:.2f}x+{b[0,0]:.2f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('Linear Regression')
axes[0].legend()

# Loss 곡선
axes[1].plot(losses)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Training Loss')

plt.tight_layout()
plt.savefig('result.png', dpi=100)
print("\n시각화 저장: result.png")
