"""
Two-Tower Retrieval Model with PyTorch.
쿼리/아이템 타워로 임베딩 기반 검색.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# --- 합성 데이터 ---
n_users = 200
n_items = 100
user_feat_dim = 8
item_feat_dim = 6
embed_dim = 16

# 유저/아이템 특성
user_features = torch.randn(n_users, user_feat_dim)
item_features = torch.randn(n_items, item_feat_dim)

# 상호작용: 유저-아이템 쌍 (positive)
n_interactions = 1000
pos_users = torch.randint(0, n_users, (n_interactions,))
pos_items = torch.randint(0, n_items, (n_interactions,))

print(f"users: {n_users}, items: {n_items}, interactions: {n_interactions}")

# --- Two-Tower 모델 ---
class UserTower(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)  # L2 normalize

class ItemTower(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim):
        super().__init__()
        self.user_tower = UserTower(user_dim, embed_dim)
        self.item_tower = ItemTower(item_dim, embed_dim)

    def forward(self, user_feat, item_feat):
        user_emb = self.user_tower(user_feat)
        item_emb = self.item_tower(item_feat)
        return user_emb, item_emb

model = TwoTower(user_feat_dim, item_feat_dim, embed_dim)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습: In-batch Negative Sampling ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 128
num_epochs = 50

for epoch in range(num_epochs):
    perm = torch.randperm(n_interactions)
    total_loss = 0
    n_batches = 0

    for i in range(0, n_interactions, batch_size):
        idx = perm[i:i+batch_size]
        u_idx = pos_users[idx]
        i_idx = pos_items[idx]

        u_feat = user_features[u_idx]
        i_feat = item_features[i_idx]

        user_emb, item_emb = model(u_feat, i_feat)

        # In-batch negative: 배치 내 모든 유저-아이템 쌍의 점수 계산
        scores = user_emb @ item_emb.T  # (B, B)
        # 대각선이 positive, 나머지가 negative
        labels = torch.arange(len(idx))
        loss = nn.CrossEntropyLoss()(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / n_batches
        print(f"epoch {epoch+1:2d} | loss {avg_loss:.4f}")

# --- Retrieval ---
print("\n--- retrieval ---")
model.eval()
with torch.no_grad():
    all_item_emb = model.item_tower(item_features)  # (n_items, embed_dim)

    for uid in [0, 10, 50]:
        user_emb = model.user_tower(user_features[uid:uid+1])  # (1, embed_dim)
        scores = (user_emb @ all_item_emb.T).squeeze()  # (n_items,)
        top5 = scores.topk(5)
        items = [f"item{idx}({s:.2f})" for idx, s in zip(top5.indices.tolist(), top5.values.tolist())]
        print(f"  user {uid}: {', '.join(items)}")
