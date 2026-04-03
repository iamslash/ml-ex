"""
PLE (Progressive Layered Extraction) Multi-Task Learning with PyTorch.
다중 과제를 공유/전용 expert 네트워크로 동시 학습.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
# Task A: 클릭 예측, Task B: 좋아요 예측
n_samples = 2000
n_features = 16

X = torch.randn(n_samples, n_features)
# Task A: 특성 0-7에 의존
y_a = (X[:, :8].sum(dim=1) > 0).float()
# Task B: 특성 4-15에 의존 (일부 공유)
y_b = (X[:, 4:].sum(dim=1) > 0).float()

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
ya_train, ya_test = y_a[:split], y_a[split:]
yb_train, yb_test = y_b[:split], y_b[split:]

print(f"samples: {n_samples}, features: {n_features}")
print(f"task A positive: {y_a.mean():.2f}, task B positive: {y_b.mean():.2f}")

# --- SwiGLU Activation ---
class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit (LLM에서 사용되는 활성화)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.W(x) * torch.sigmoid(self.V(x))

# --- Expert Network ---
class Expert(nn.Module):
    def __init__(self, in_dim, out_dim, use_swiglu=True):
        super().__init__()
        if use_swiglu:
            self.net = nn.Sequential(
                SwiGLU(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim),
            )

    def forward(self, x):
        return self.net(x)

# --- Gating Network ---
class Gate(nn.Module):
    def __init__(self, in_dim, n_experts):
        super().__init__()
        self.gate = nn.Linear(in_dim, n_experts)

    def forward(self, x, expert_outputs):
        # expert_outputs: list of (batch, expert_dim)
        weights = torch.softmax(self.gate(x), dim=1)  # (batch, n_experts)
        stacked = torch.stack(expert_outputs, dim=1)   # (batch, n_experts, expert_dim)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (batch, expert_dim)

# --- PLE Layer ---
class PLELayer(nn.Module):
    def __init__(self, in_dim, expert_dim, n_shared, n_task_experts, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_shared = n_shared
        self.n_task_experts = n_task_experts
        total_experts = n_shared + n_task_experts

        # 공유 expert
        self.shared_experts = nn.ModuleList([Expert(in_dim, expert_dim) for _ in range(n_shared)])

        # 태스크별 전용 expert
        self.task_experts = nn.ModuleList([
            nn.ModuleList([Expert(in_dim, expert_dim) for _ in range(n_task_experts)])
            for _ in range(n_tasks)
        ])

        # 태스크별 gate
        self.gates = nn.ModuleList([Gate(in_dim, total_experts) for _ in range(n_tasks)])

    def forward(self, x):
        # 공유 expert 출력
        shared_outs = [e(x) for e in self.shared_experts]

        task_outputs = []
        for t in range(self.n_tasks):
            # 태스크 전용 expert 출력
            task_outs = [e(x) for e in self.task_experts[t]]
            # 공유 + 전용 expert 결합
            all_outs = shared_outs + task_outs
            # Gate로 가중 합산
            gated = self.gates[t](x, all_outs)
            task_outputs.append(gated)

        return task_outputs

# --- PLE 모델 ---
class PLE(nn.Module):
    def __init__(self, in_dim, expert_dim=32, n_shared=2, n_task_experts=2, n_tasks=2, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in = in_dim if i == 0 else expert_dim
            self.layers.append(PLELayer(layer_in, expert_dim, n_shared, n_task_experts, n_tasks))

        # 태스크별 타워
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        task_inputs = None
        for layer in self.layers:
            if task_inputs is None:
                task_inputs = layer(x)
            else:
                task_inputs = [layer(ti)[i] for i, ti in enumerate(task_inputs)]

        outputs = [self.towers[i](task_inputs[i]).squeeze() for i in range(len(self.towers))]
        return outputs

# --- Uncertainty Loss ---
class UncertaintyLoss(nn.Module):
    """학습 가능한 태스크 가중치 (Kendall et al.)"""
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total

model = PLE(n_features, expert_dim=32, n_shared=2, n_task_experts=2, n_tasks=2, n_layers=2)
unc_loss = UncertaintyLoss(2)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
bce = nn.BCEWithLogitsLoss()
all_params = list(model.parameters()) + list(unc_loss.parameters())
optimizer = optim.Adam(all_params, lr=0.001)
batch_size = 128

for epoch in range(50):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        out_a, out_b = model(X_train[idx])

        loss_a = bce(out_a, ya_train[idx])
        loss_b = bce(out_b, yb_train[idx])
        loss = unc_loss([loss_a, loss_b])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            ta, tb = model(X_test)
            acc_a = ((ta > 0).float() == ya_test).float().mean().item() * 100
            acc_b = ((tb > 0).float() == yb_test).float().mean().item() * 100
            weights = torch.exp(-unc_loss.log_vars).detach()
        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | "
              f"task A {acc_a:.1f}% | task B {acc_b:.1f}% | "
              f"weights [{weights[0]:.2f}, {weights[1]:.2f}]")
