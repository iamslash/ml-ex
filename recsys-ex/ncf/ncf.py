"""
Neural Collaborative Filtering with PyTorch.
MLP 기반 유저-아이템 상호작용 예측.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
n_users = 100
n_items = 50
embed_dim = 16

# 상호작용 생성 (implicit feedback)
n_pos = 800
pos_users = torch.randint(0, n_users, (n_pos,))
pos_items = torch.randint(0, n_items, (n_pos,))

# Negative sampling
n_neg = n_pos
neg_users = torch.randint(0, n_users, (n_neg,))
neg_items = torch.randint(0, n_items, (n_neg,))

users = torch.cat([pos_users, neg_users])
items = torch.cat([pos_items, neg_items])
labels = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])

# shuffle
perm = torch.randperm(len(labels))
users, items, labels = users[perm], items[perm], labels[perm]

split = int(0.8 * len(labels))
train_u, test_u = users[:split], users[split:]
train_i, test_i = items[:split], items[split:]
train_y, test_y = labels[:split], labels[split:]

print(f"users: {n_users}, items: {n_items}")
print(f"train: {split}, test: {len(labels) - split}")

# --- NCF 모델 ---
class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, mlp_dims):
        super().__init__()
        # GMF 경로
        self.gmf_user_emb = nn.Embedding(n_users, embed_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embed_dim)

        # MLP 경로
        self.mlp_user_emb = nn.Embedding(n_users, embed_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embed_dim)

        layers = []
        in_dim = embed_dim * 2
        for dim in mlp_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        self.mlp = nn.Sequential(*layers)

        # 최종 예측
        self.fc_out = nn.Linear(embed_dim + mlp_dims[-1], 1)

    def forward(self, user_ids, item_ids):
        # GMF: element-wise product
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i  # (B, embed_dim)

        # MLP: concatenate + feedforward
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=1)  # (B, embed_dim*2)
        mlp_out = self.mlp(mlp_input)  # (B, mlp_dims[-1])

        # Combine
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.fc_out(combined).squeeze()

model = NCF(n_users, n_items, embed_dim, mlp_dims=[32, 16])
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 128
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        pred = model(train_u[idx], train_i[idx])
        loss = criterion(pred, train_y[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(test_u, test_i)
            test_loss = criterion(test_pred, test_y).item()
            test_acc = ((test_pred > 0).float() == test_y).float().mean().item() * 100
        print(f"epoch {epoch+1:2d} | train loss {total_loss/n_batches:.4f} | test loss {test_loss:.4f} | test acc {test_acc:.1f}%")

# --- 추천 생성 ---
print("\n--- recommendations ---")
model.eval()
with torch.no_grad():
    for uid in [0, 10, 25]:
        all_items = torch.arange(n_items)
        all_users = torch.full_like(all_items, uid)
        scores = model(all_users, all_items)
        probs = torch.sigmoid(scores)
        top5 = probs.topk(5)
        items = [f"item{idx}({s:.2f})" for idx, s in zip(top5.indices.tolist(), top5.values.tolist())]
        print(f"  user {uid}: {', '.join(items)}")
