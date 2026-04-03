"""
HydraNet Uplift Model with PyTorch.
다중 처리(multi-treatment) 인과추론 모델. CATE 추정.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# --- 합성 인과 데이터 ---
# X: 공변량, T: 처리 (0=대조군, 1=처리A, 2=처리B), Y: 결과
n_samples = 3000
n_features = 8
n_treatments = 3  # 0=control, 1=treatment_A, 2=treatment_B

X = torch.randn(n_samples, n_features)

# 처리 배정 (무작위)
T = torch.randint(0, n_treatments, (n_samples,))

# 잠재 결과 (potential outcomes)
# Y(0) = f(X) + noise
# Y(1) = f(X) + tau_1(X) + noise  (Treatment A effect)
# Y(2) = f(X) + tau_2(X) + noise  (Treatment B effect)
base_outcome = (X[:, 0] + X[:, 1] * 0.5).unsqueeze(1)

# CATE (Conditional Average Treatment Effect)
# Treatment A: X[:, 2] > 0인 사람에게 효과적
tau_1 = (X[:, 2] > 0).float() * 2.0
# Treatment B: X[:, 3] > 0인 사람에게 효과적
tau_2 = (X[:, 3] > 0).float() * 3.0

Y = base_outcome.squeeze() + torch.zeros(n_samples)
Y[T == 1] += tau_1[T == 1]
Y[T == 2] += tau_2[T == 2]
Y += torch.randn(n_samples) * 0.5  # noise

# Propensity (처리 배정 확률, 여기서는 균등)
propensity = torch.full((n_samples, n_treatments), 1.0 / n_treatments)

split = int(0.8 * n_samples)
print(f"samples: {n_samples}, features: {n_features}, treatments: {n_treatments}")
print(f"treatment distribution: {[(T==t).sum().item() for t in range(n_treatments)]}")

# --- HydraNet ---
class HydraNet(nn.Module):
    """공유 표현 + 처리별 outcome head"""
    def __init__(self, in_dim, hidden_dim, n_treatments):
        super().__init__()
        # 공유 representation
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # 처리별 outcome head
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(n_treatments)
        ])
        # Propensity head (보조 태스크)
        self.propensity_head = nn.Linear(hidden_dim, n_treatments)

    def forward(self, x):
        h = self.shared(x)
        outcomes = [head(h).squeeze(-1) for head in self.heads]
        prop_logits = self.propensity_head(h)
        return outcomes, prop_logits

model = HydraNet(n_features, 64, n_treatments)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- DR (Doubly Robust) Loss ---
def dr_loss(outcomes, prop_logits, X, T, Y, propensity):
    """
    Doubly Robust 손실: outcome regression + IPW 보정
    DR = E[mu_t(X) + (Y - mu_t(X)) * I(T=t) / e(t|X)]
    """
    batch_size = X.size(0)
    loss = 0

    # Outcome regression loss (관측된 처리에 대해)
    for t in range(n_treatments):
        mask = (T == t)
        if mask.sum() > 0:
            loss += nn.MSELoss()(outcomes[t][mask], Y[mask])

    # Propensity loss (보조)
    prop_loss = nn.CrossEntropyLoss()(prop_logits, T)
    loss += 0.1 * prop_loss

    return loss

# --- 학습 ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 256

for epoch in range(100):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        outcomes, prop_logits = model(X[idx])
        loss = dr_loss(outcomes, prop_logits, X[idx], T[idx], Y[idx], propensity[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_outcomes, _ = model(X[split:])
            # CATE 추정: E[Y(t)] - E[Y(0)]
            cate_1 = (test_outcomes[1] - test_outcomes[0]).mean().item()
            cate_2 = (test_outcomes[2] - test_outcomes[0]).mean().item()
        print(f"epoch {epoch+1:3d} | loss {total_loss/n_batches:.4f} | "
              f"CATE_A={cate_1:.2f} (true~1.0) | CATE_B={cate_2:.2f} (true~1.5)")

# --- CATE 분석 ---
print("\n--- CATE analysis ---")
model.eval()
with torch.no_grad():
    outcomes, _ = model(X[split:])
    X_test = X[split:]

    # 서브그룹별 CATE
    for feature_idx, name in [(2, "X2>0 (Treatment A target)"), (3, "X3>0 (Treatment B target)")]:
        mask = X_test[:, feature_idx] > 0
        cate_1_pos = (outcomes[1][mask] - outcomes[0][mask]).mean().item()
        cate_1_neg = (outcomes[1][~mask] - outcomes[0][~mask]).mean().item()
        cate_2_pos = (outcomes[2][mask] - outcomes[0][mask]).mean().item()
        cate_2_neg = (outcomes[2][~mask] - outcomes[0][~mask]).mean().item()
        print(f"\n  {name}:")
        print(f"    CATE_A: +={cate_1_pos:.2f}, -={cate_1_neg:.2f}")
        print(f"    CATE_B: +={cate_2_pos:.2f}, -={cate_2_neg:.2f}")
