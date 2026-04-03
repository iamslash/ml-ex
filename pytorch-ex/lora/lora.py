"""
LoRA (Low-Rank Adaptation) with PyTorch.
MLP에 LoRA를 적용하여 파라미터 효율적 fine-tuning 시연.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# --- LoRA 레이어 ---
class LoRALinear(nn.Module):
    """기존 Linear에 low-rank adapter를 추가"""
    def __init__(self, original_linear, rank=4):
        super().__init__()
        self.original = original_linear
        # 원래 가중치는 동결
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Low-rank 분해: W + A @ B (rank << min(in, out))
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        # 원래 출력 + low-rank 보정
        return self.original(x) + x @ self.lora_A @ self.lora_B

# --- 기본 모델 (사전학습된 모델 역할) ---
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- 데이터 생성 ---
# 사전학습 데이터: y = sin(x)
n = 500
X_pretrain = torch.randn(n, 4)
y_pretrain = torch.sin(X_pretrain).sum(dim=1, keepdim=True)

# Fine-tune 데이터: y = cos(x) (다른 태스크)
X_finetune = torch.randn(200, 4)
y_finetune = torch.cos(X_finetune).sum(dim=1, keepdim=True)

pretrain_loader = DataLoader(TensorDataset(X_pretrain, y_pretrain), batch_size=64, shuffle=True)
finetune_loader = DataLoader(TensorDataset(X_finetune, y_finetune), batch_size=64, shuffle=True)

# --- Step 1: 사전학습 ---
print("=== Step 1: Pretrain ===")
model = MLP(4, 64, 1)
total_params = sum(p.numel() for p in model.parameters())
print(f"total params: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(50):
    for X_batch, y_batch in pretrain_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            pred = model(X_pretrain)
            loss = criterion(pred, y_pretrain)
        print(f"  epoch {epoch+1:2d} | pretrain loss {loss:.4f}")

# --- Step 2: LoRA 적용 ---
print("\n=== Step 2: Apply LoRA ===")
rank = 4

# 기존 Linear를 LoRA Linear로 교체
model.fc1 = LoRALinear(model.fc1, rank=rank)
model.fc2 = LoRALinear(model.fc2, rank=rank)

# 학습 가능한 파라미터 확인
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable params: {trainable} ({trainable/total_params*100:.1f}%)")
print(f"frozen params: {frozen}")

# --- Step 3: LoRA Fine-tuning ---
print("\n=== Step 3: LoRA Fine-tune ===")
# LoRA 파라미터만 최적화
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(lora_params, lr=0.01)

for epoch in range(50):
    for X_batch, y_batch in finetune_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            pred = model(X_finetune)
            loss = criterion(pred, y_finetune)
        print(f"  epoch {epoch+1:2d} | finetune loss {loss:.4f}")

# --- 비교 ---
print("\n=== Comparison ===")
with torch.no_grad():
    finetune_loss = criterion(model(X_finetune), y_finetune)
    pretrain_loss = criterion(model(X_pretrain), y_pretrain)

print(f"finetune task loss: {finetune_loss:.4f}")
print(f"pretrain task loss: {pretrain_loss:.4f}")
print(f"\nLoRA rank={rank}: {trainable} trainable params out of {total_params} total ({trainable/total_params*100:.1f}%)")
