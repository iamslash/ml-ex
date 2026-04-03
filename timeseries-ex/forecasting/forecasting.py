"""
Time Series Forecasting with PyTorch.
LSTM 기반 시계열 예측.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# --- 합성 시계열 데이터 ---
# sin + trend + noise
n_points = 500
t = np.arange(n_points, dtype=np.float32)
series = np.sin(t * 0.1) + 0.02 * t + 0.3 * np.random.randn(n_points)
series = (series - series.mean()) / series.std()  # 정규화

print(f"series length: {n_points}")

# --- 윈도우 생성 ---
window_size = 20
pred_len = 1  # 다음 1스텝 예측

X, y = [], []
for i in range(len(series) - window_size - pred_len + 1):
    X.append(series[i:i+window_size])
    y.append(series[i+window_size:i+window_size+pred_len])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, window, 1)
y = torch.tensor(np.array(y), dtype=torch.float32).squeeze(-1)    # (N,)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"windows: {len(X)}, train: {split}, test: {len(X)-split}")

# --- LSTM 모델 ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, window, 1)
        output, (h_n, c_n) = self.lstm(x)
        last = output[:, -1, :]  # (batch, hidden)
        return self.fc(last).squeeze(-1)  # (batch,)

model = LSTMForecaster()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64

for epoch in range(100):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        pred = model(X_train[idx])
        loss = criterion(pred, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
        print(f"epoch {epoch+1:3d} | train MSE {total_loss/n_batches:.6f} | test MSE {test_loss.item():.6f}")

# --- 평가 ---
model.eval()
with torch.no_grad():
    test_pred = model(X_test)

mae = (test_pred - y_test).abs().mean().item()
rmse = ((test_pred - y_test) ** 2).mean().sqrt().item()
print(f"\ntest MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 방향 정확도 (상승/하락 맞춤)
actual_dir = (y_test[1:] - y_test[:-1]) > 0
pred_dir = (test_pred[1:] - test_pred[:-1]) > 0
dir_acc = (actual_dir == pred_dir).float().mean().item() * 100
print(f"direction accuracy: {dir_acc:.1f}%")
