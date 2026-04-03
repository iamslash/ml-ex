"""
GAT (Graph Attention Network) with PyTorch.
어텐션 가중치로 이웃 노드를 집계하는 그래프 신경망.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)

# --- 합성 그래프 데이터 (GCN과 동일 구조) ---
n_nodes = 30
n_classes = 3
feat_dim = 8

node_features = torch.randn(n_nodes, feat_dim)
for c in range(n_classes):
    start, end = c * 10, (c + 1) * 10
    center = torch.randn(feat_dim)
    node_features[start:end] += center * 2

labels = torch.tensor([i // 10 for i in range(n_nodes)])

# 인접 행렬
adj = torch.zeros(n_nodes, n_nodes)
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        same = (i // 10) == (j // 10)
        if torch.rand(1).item() < (0.4 if same else 0.05):
            adj[i, j] = 1
            adj[j, i] = 1
adj += torch.eye(n_nodes)

# 엣지 리스트 (sparse representation)
edge_index = adj.nonzero(as_tuple=False).T  # (2, n_edges)
n_edges = int((adj.sum() - n_nodes) / 2)

train_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
for c in range(n_classes):
    idx = (labels == c).nonzero().squeeze()
    perm = idx[torch.randperm(len(idx))]
    train_mask[perm[:7]] = True
    test_mask[perm[7:]] = True

print(f"nodes: {n_nodes}, edges: {n_edges}, classes: {n_classes}")
print(f"train: {train_mask.sum()}, test: {test_mask.sum()}")

# --- GAT Layer ---
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        # 어텐션 벡터 a = [a_left || a_right]
        self.a_left = nn.Parameter(torch.randn(n_heads, out_dim) * 0.01)
        self.a_right = nn.Parameter(torch.randn(n_heads, out_dim) * 0.01)

    def forward(self, x, edge_index):
        N = x.size(0)
        # Linear projection
        h = self.W(x).view(N, self.n_heads, self.out_dim)  # (N, heads, out_dim)

        src, dst = edge_index  # (n_edges,), (n_edges,)

        # Attention scores: e_ij = LeakyReLU(a_left * h_i + a_right * h_j)
        h_src = h[src]  # (n_edges, heads, out_dim)
        h_dst = h[dst]  # (n_edges, heads, out_dim)

        e = (h_src * self.a_left).sum(dim=-1) + (h_dst * self.a_right).sum(dim=-1)  # (n_edges, heads)
        e = F.leaky_relu(e, 0.2)

        # Softmax per destination node
        # Sparse softmax: group by dst
        alpha = torch.zeros(N, self.n_heads, device=x.device)
        e_exp = e.exp()
        # Sum of exp for each dst
        denom = torch.zeros(N, self.n_heads, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (denom[dst] + 1e-8)  # (n_edges, heads)

        # Weighted sum of neighbor features
        out = torch.zeros(N, self.n_heads, self.out_dim, device=x.device)
        weighted = h_src * alpha.unsqueeze(-1)  # (n_edges, heads, out_dim)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted)

        if self.concat:
            return out.view(N, -1)  # (N, heads * out_dim)
        else:
            return out.mean(dim=1)  # (N, out_dim)

# --- GAT 모델 ---
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=4):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, n_heads, concat=True)
        self.gat2 = GATLayer(hidden_dim * n_heads, out_dim, 1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

model = GAT(feat_dim, 8, n_classes, n_heads=4)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    output = model(node_features, edge_index)
    loss = criterion(output[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(node_features, edge_index).argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item() * 100
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | train {train_acc:.1f}% | test {test_acc:.1f}%")

# --- GCN과의 비교 ---
print("\n--- GAT vs GCN ---")
print("  GCN: 모든 이웃을 동일 가중치로 집계 (degree normalization)")
print("  GAT: 이웃마다 어텐션 가중치를 학습하여 중요한 이웃에 더 집중")
