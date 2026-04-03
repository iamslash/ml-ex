"""
Anomaly Detection with Autoencoder (PyTorch).
시계열 데이터에서 reconstruction error로 이상 탐지.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# --- 합성 시계열 데이터 ---
n_normal = 500
n_anomaly = 50
window_size = 20

# 정상: sin 패턴
t = np.linspace(0, 10 * np.pi, n_normal * window_size)
normal_signal = np.sin(t) + 0.1 * np.random.randn(len(t))
normal_windows = normal_signal[:n_normal * window_size].reshape(n_normal, window_size)

# 이상: 스파이크, 드리프트
anomaly_windows = []
for _ in range(n_anomaly):
    w = np.sin(np.linspace(0, 2 * np.pi, window_size)) + 0.1 * np.random.randn(window_size)
    anomaly_type = np.random.choice(['spike', 'drift', 'flat'])
    if anomaly_type == 'spike':
        w[np.random.randint(0, window_size)] += np.random.choice([-5, 5])
    elif anomaly_type == 'drift':
        w += np.linspace(0, 3, window_size)
    else:
        w = np.ones(window_size) * np.random.randn()
    anomaly_windows.append(w)
anomaly_windows = np.array(anomaly_windows)

X_normal = torch.tensor(normal_windows, dtype=torch.float32)
X_anomaly = torch.tensor(anomaly_windows, dtype=torch.float32)

# 정상 데이터로만 학습
split = int(0.8 * n_normal)
X_train = X_normal[:split]
X_val = X_normal[split:]

print(f"train (normal): {len(X_train)}, val (normal): {len(X_val)}, anomaly: {len(X_anomaly)}")
print(f"window size: {window_size}")

# --- Autoencoder ---
class AnomalyAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = AnomalyAE(window_size)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 정상 데이터로만 학습 ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    x_recon = model(X_train)
    loss = criterion(x_recon, X_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss = criterion(val_recon, X_val)
        print(f"epoch {epoch+1:3d} | train loss {loss.item():.6f} | val loss {val_loss.item():.6f}")

# --- 이상 탐지 ---
model.eval()
with torch.no_grad():
    normal_errors = ((model(X_val) - X_val) ** 2).mean(dim=1)
    anomaly_errors = ((model(X_anomaly) - X_anomaly) ** 2).mean(dim=1)

print(f"\n--- reconstruction errors ---")
print(f"normal  - mean: {normal_errors.mean():.4f}, std: {normal_errors.std():.4f}")
print(f"anomaly - mean: {anomaly_errors.mean():.4f}, std: {anomaly_errors.std():.4f}")

# 임계값 설정: 정상 데이터의 mean + 2*std
threshold = normal_errors.mean() + 2 * normal_errors.std()
print(f"\nthreshold (mean + 2*std): {threshold:.4f}")

normal_detected = (normal_errors > threshold).sum().item()
anomaly_detected = (anomaly_errors > threshold).sum().item()

print(f"normal  flagged as anomaly: {normal_detected}/{len(X_val)} (FPR: {normal_detected/len(X_val)*100:.1f}%)")
print(f"anomaly detected:          {anomaly_detected}/{len(X_anomaly)} (Recall: {anomaly_detected/len(X_anomaly)*100:.1f}%)")
