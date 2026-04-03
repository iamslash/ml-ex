"""
Deep & Cross Network (DCN-v2) with PyTorch.
명시적 feature interaction을 자동으로 학습하는 추천 랭킹 모델.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
# CTR 예측: 유저/아이템 특성 → 클릭 확률
n_samples = 2000
n_features = 16

X = torch.randn(n_samples, n_features)
# 정답: 2차 교차항 + 비선형 결합
y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + torch.sin(X[:, 4]) > 0).float()

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"samples: {n_samples}, features: {n_features}")
print(f"train: {split}, test: {n_samples - split}")
print(f"positive ratio: {y.mean():.2f}")

# --- Cross Layer (DCN-v2) ---
class CrossLayer(nn.Module):
    """x_{l+1} = x_0 * (W @ x_l + b) + x_l"""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, x0, xl):
        return x0 * self.W(xl) + xl

# --- DCN-v2 모델 ---
class DCNv2(nn.Module):
    def __init__(self, in_dim, cross_layers=3, deep_dims=[64, 32]):
        super().__init__()
        # Cross Network
        self.cross_layers = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_layers)])

        # Deep Network
        layers = []
        prev_dim = in_dim
        for dim in deep_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.deep = nn.Sequential(*layers)

        # Combine
        self.fc_out = nn.Linear(in_dim + deep_dims[-1], 1)

    def forward(self, x):
        # Cross path
        x0 = x
        xl = x
        for cross in self.cross_layers:
            xl = cross(x0, xl)

        # Deep path
        deep_out = self.deep(x)

        # Combine
        combined = torch.cat([xl, deep_out], dim=1)
        return self.fc_out(combined).squeeze()

model = DCNv2(n_features, cross_layers=3, deep_dims=[64, 32])
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 128

for epoch in range(50):
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

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_acc = ((test_pred > 0).float() == y_test).float().mean().item() * 100
            test_loss = criterion(test_pred, y_test).item()
        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | test loss {test_loss:.4f} | test acc {test_acc:.1f}%")

# --- Cross Layer 분석 ---
print("\n--- cross layer weight norms ---")
for i, cross in enumerate(model.cross_layers):
    w_norm = cross.W.weight.norm().item()
    print(f"  layer {i}: W norm = {w_norm:.4f}")
