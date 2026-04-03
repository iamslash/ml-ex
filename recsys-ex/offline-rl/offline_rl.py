"""
Offline RL with DQN + CQL (Conservative Q-Learning) with PyTorch.
오프라인 데이터만으로 추천 정책을 학습.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# --- 합성 오프라인 데이터 ---
# 상태: 유저 특성 (8차원)
# 행동: 추천 아이템 선택 (10개 중 1개)
# 보상: 클릭(+1), 미클릭(0), 이탈(-1)
state_dim = 8
n_actions = 10
n_transitions = 5000

states = torch.randn(n_transitions, state_dim)
actions = torch.randint(0, n_actions, (n_transitions,))
# 보상: 특정 상태-행동 조합에 높은 보상
rewards = torch.zeros(n_transitions)
for i in range(n_transitions):
    # 최적 행동: 상태의 argmax와 행동이 일치하면 +1
    optimal = states[i, :n_actions].argmax().item()
    if actions[i].item() == optimal:
        rewards[i] = 1.0
    elif torch.rand(1).item() < 0.1:
        rewards[i] = -1.0

next_states = states + 0.1 * torch.randn_like(states)
dones = (torch.rand(n_transitions) < 0.05).float()

print(f"transitions: {n_transitions}, state_dim: {state_dim}, actions: {n_actions}")
print(f"reward distribution: +1={int((rewards==1).sum())}, 0={int((rewards==0).sum())}, -1={int((rewards==-1).sum())}")

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, state):
        return self.net(state)

# --- CQL Loss ---
def cql_loss(q_values, actions, alpha=1.0):
    """
    Conservative Q-Learning: 분포 외 행동의 Q값을 페널티
    L_CQL = alpha * (log_sum_exp(Q(s,a)) - Q(s, a_data))
    """
    # log_sum_exp: 모든 행동의 Q값 (과대추정 방지)
    logsumexp = torch.logsumexp(q_values, dim=1).mean()
    # 데이터에 있는 행동의 Q값
    data_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
    return alpha * (logsumexp - data_q)

# --- 학습 ---
q_net = QNetwork(state_dim, n_actions)
target_net = QNetwork(state_dim, n_actions)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.001)
gamma = 0.99
cql_alpha = 0.5
batch_size = 256
num_epochs = 100

for epoch in range(num_epochs):
    perm = torch.randperm(n_transitions)[:batch_size]

    s = states[perm]
    a = actions[perm]
    r = rewards[perm]
    s_next = next_states[perm]
    d = dones[perm]

    # Current Q
    q_values = q_net(s)
    q_a = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

    # Target Q
    with torch.no_grad():
        q_next = target_net(s_next)
        q_target = r + gamma * q_next.max(dim=1)[0] * (1 - d)

    # TD loss
    td_loss = nn.MSELoss()(q_a, q_target)

    # CQL penalty
    cql = cql_loss(q_values, a, alpha=cql_alpha)

    # Total loss
    loss = td_loss + cql

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Target network update
    if (epoch + 1) % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())

    if (epoch + 1) % 20 == 0:
        # 정책 평가
        with torch.no_grad():
            all_q = q_net(states)
            policy_actions = all_q.argmax(dim=1)
            optimal_actions = states[:, :n_actions].argmax(dim=1)
            policy_acc = (policy_actions == optimal_actions).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | td_loss {td_loss.item():.4f} | cql {cql.item():.4f} | policy acc {policy_acc:.1f}%")

# --- 정책 비교 ---
print("\n--- policy comparison ---")
with torch.no_grad():
    all_q = q_net(states)
    learned_actions = all_q.argmax(dim=1)
    optimal_actions = states[:, :n_actions].argmax(dim=1)

    # 학습된 정책
    learned_reward = rewards[learned_actions == actions].mean().item()
    # 데이터 정책 (행동 정책)
    data_reward = rewards.mean().item()
    # 최적 정책
    optimal_mask = (optimal_actions == actions)
    optimal_reward = rewards[optimal_mask].mean().item() if optimal_mask.sum() > 0 else 0

    print(f"  data policy avg reward:    {data_reward:.4f}")
    print(f"  learned policy accuracy:   {(learned_actions == optimal_actions).float().mean().item()*100:.1f}%")

    # CQL vs 일반 DQN 비교 설명
    print(f"\n--- why CQL? ---")
    print(f"  standard DQN: Q값을 과대추정 → 분포 외 행동을 선택 → 실패")
    print(f"  CQL: 분포 외 행동의 Q값에 페널티 → 보수적 정책 → 안전한 배포")
