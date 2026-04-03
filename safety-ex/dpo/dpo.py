"""
DPO (Direct Preference Optimization) with PyTorch.
Reward model 없이 직접 선호도로 정책 최적화.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
# 입력(프롬프트) + chosen/rejected 응답을 특성 벡터로 표현
feat_dim = 8
n_pairs = 200

# 프롬프트 특성
prompts = torch.randn(n_pairs, feat_dim)

# chosen은 특정 방향으로, rejected는 다른 방향으로
true_direction = torch.randn(feat_dim)
true_direction = true_direction / true_direction.norm()

chosen = prompts + 0.5 * true_direction + 0.1 * torch.randn(n_pairs, feat_dim)
rejected = prompts - 0.3 * true_direction + 0.1 * torch.randn(n_pairs, feat_dim)

split = int(0.8 * n_pairs)
print(f"pairs: {n_pairs}, train: {split}, test: {n_pairs - split}")

# --- 정책 모델 (학습 대상) ---
class PolicyModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

policy = PolicyModel(feat_dim)

# 참조 모델 (동결, DPO에서 KL 발산 계산용)
ref_policy = PolicyModel(feat_dim)
ref_policy.load_state_dict(policy.state_dict())
for param in ref_policy.parameters():
    param.requires_grad = False

print(f"params: {sum(p.numel() for p in policy.parameters()):,}")

# --- DPO Loss ---
# L_DPO = -log(sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))
beta = 0.1

optimizer = optim.Adam(policy.parameters(), lr=0.001)

for epoch in range(200):
    policy.train()

    # 학습 데이터
    p_chosen = policy(chosen[:split])
    p_rejected = policy(rejected[:split])

    with torch.no_grad():
        ref_chosen = ref_policy(chosen[:split])
        ref_rejected = ref_policy(rejected[:split])

    # Log ratio
    log_ratio_chosen = p_chosen - ref_chosen
    log_ratio_rejected = p_rejected - ref_rejected

    # DPO loss
    loss = -torch.log(torch.sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)) + 1e-8).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        # 평가
        policy.eval()
        with torch.no_grad():
            tc = policy(chosen[split:])
            tr = policy(rejected[split:])
            test_acc = (tc > tr).float().mean().item() * 100

            train_c = policy(chosen[:split])
            train_r = policy(rejected[:split])
            train_acc = (train_c > train_r).float().mean().item() * 100

        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | train acc {train_acc:.1f}% | test acc {test_acc:.1f}%")

# --- Reward Model과의 비교 ---
print("\n--- DPO vs Reward Model ---")
print("DPO: 별도 reward model 없이 정책을 직접 최적화")
print("RLHF: reward model 학습 -> RL로 정책 최적화 (2단계)")

policy.eval()
with torch.no_grad():
    c_scores = policy(chosen[split:])
    r_scores = policy(rejected[split:])
    margin = (c_scores - r_scores).mean().item()
    acc = (c_scores > r_scores).float().mean().item() * 100
    print(f"\nfinal: chosen-rejected margin = {margin:.3f}, accuracy = {acc:.1f}%")
