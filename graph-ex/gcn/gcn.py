"""
GCN (Graph Convolutional Network) with PyTorch.
노드 분류 과제.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# --- 합성 그래프 데이터 ---
# Karate Club 유사: 3개 커뮤니티, 각 10개 노드
n_nodes = 30
n_classes = 3
feat_dim = 8

# 노드 특성
node_features = torch.randn(n_nodes, feat_dim)
# 같은 커뮤니티 노드는 비슷한 특성
for c in range(n_classes):
    start, end = c * 10, (c + 1) * 10
    center = torch.randn(feat_dim)
    node_features[start:end] += center * 2

labels = torch.tensor([i // 10 for i in range(n_nodes)])

# 인접 행렬 (같은 커뮤니티 내 연결 확률 높음)
adj = torch.zeros(n_nodes, n_nodes)
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        same_community = (i // 10) == (j // 10)
        prob = 0.4 if same_community else 0.05
        if torch.rand(1).item() < prob:
            adj[i, j] = 1
            adj[j, i] = 1

# Self-loop 추가
adj = adj + torch.eye(n_nodes)

# Degree normalization: D^{-1/2} A D^{-1/2}
degree = adj.sum(dim=1)
D_inv_sqrt = torch.diag(1.0 / degree.sqrt())
adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

n_edges = int((adj.sum() - n_nodes) / 2)
print(f"nodes: {n_nodes}, edges: {n_edges}, classes: {n_classes}")

# Train/test split
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
for c in range(n_classes):
    idx = (labels == c).nonzero().squeeze()
    perm = idx[torch.randperm(len(idx))]
    train_mask[perm[:7]] = True
    test_mask[perm[7:]] = True

print(f"train: {train_mask.sum()}, test: {test_mask.sum()}")

# --- GCN Layer ---
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # Message passing: A_norm @ X @ W
        return self.linear(adj @ x)

# --- GCN 모델 ---
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

model = GCN(feat_dim, 16, n_classes)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    output = model(node_features, adj_norm)
    loss = criterion(output[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(node_features, adj_norm).argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item() * 100
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | train {train_acc:.1f}% | test {test_acc:.1f}%")

# --- 결과 ---
print("\n--- node classification ---")
model.eval()
with torch.no_grad():
    pred = model(node_features, adj_norm).argmax(dim=1)
    for c in range(n_classes):
        mask = labels == c
        acc = (pred[mask] == c).float().mean().item() * 100
        print(f"  community {c}: {acc:.1f}%")
