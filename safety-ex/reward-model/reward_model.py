"""
Reward Model with PyTorch.
인간 선호도 데이터로 보상 모델 학습 (RLHF 기초).
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 선호도 데이터 ---
# 각 쌍: (chosen response, rejected response)
# 실제로는 인간 평가자가 라벨링
preference_pairs = [
    # (좋은 응답, 나쁜 응답) - 특성 벡터로 표현
    # 특성: [도움됨, 정확함, 안전함, 간결함, 공손함]
    ([0.9, 0.8, 1.0, 0.7, 0.9], [0.3, 0.2, 0.1, 0.5, 0.1]),  # 명확한 차이
    ([0.8, 0.9, 0.9, 0.6, 0.8], [0.4, 0.3, 0.8, 0.7, 0.3]),  # 안전성은 비슷
    ([0.7, 0.7, 1.0, 0.8, 0.7], [0.6, 0.6, 0.2, 0.9, 0.6]),  # 안전성 차이
    ([0.9, 0.9, 0.8, 0.5, 0.9], [0.8, 0.8, 0.3, 0.6, 0.8]),  # 미묘한 차이
    ([0.6, 0.8, 1.0, 0.7, 0.6], [0.5, 0.7, 0.4, 0.8, 0.5]),  # 안전성 차이
]

# 데이터 증강: 작은 노이즈 추가
augmented = []
for chosen, rejected in preference_pairs:
    for _ in range(20):
        noise_c = [c + torch.randn(1).item() * 0.1 for c in chosen]
        noise_r = [r + torch.randn(1).item() * 0.1 for r in rejected]
        augmented.append((noise_c, noise_r))

chosen_data = torch.tensor([c for c, r in augmented])
rejected_data = torch.tensor([r for c, r in augmented])

split = int(0.8 * len(augmented))
train_c, test_c = chosen_data[:split], chosen_data[split:]
train_r, test_r = rejected_data[:split], rejected_data[split:]

print(f"features: 5, pairs: {len(augmented)}")
print(f"train: {split}, test: {len(augmented)-split}")

# --- Reward Model ---
class RewardModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = RewardModel(5)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- Bradley-Terry Loss ---
# P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
# Loss = -log(sigmoid(r_chosen - r_rejected))
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    r_chosen = model(train_c)
    r_rejected = model(train_r)

    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        acc = (r_chosen > r_rejected).float().mean().item() * 100
        model.eval()
        with torch.no_grad():
            test_rc = model(test_c)
            test_rr = model(test_r)
            test_acc = (test_rc > test_rr).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | train acc {acc:.1f}% | test acc {test_acc:.1f}%")

# --- 보상 점수 분석 ---
print("\n--- reward scores ---")
model.eval()
with torch.no_grad():
    samples = [
        ("helpful safe", [0.9, 0.9, 1.0, 0.7, 0.9]),
        ("helpful unsafe", [0.9, 0.9, 0.1, 0.7, 0.9]),
        ("unhelpful safe", [0.1, 0.1, 1.0, 0.7, 0.9]),
        ("unhelpful unsafe", [0.1, 0.1, 0.1, 0.7, 0.1]),
    ]
    for name, feat in samples:
        score = model(torch.tensor([feat])).item()
        print(f"  {name:20s} -> reward: {score:.3f}")
