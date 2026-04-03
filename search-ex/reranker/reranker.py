"""
Cross-Encoder Re-ranker with PyTorch.
1차 검색 결과를 정밀 랭킹하는 cross-encoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
# 쿼리-문서 쌍의 관련도 점수 예측
feat_dim = 16  # 쿼리+문서 결합 특성
n_pairs = 2000

# 쿼리/문서 특성
query_feats = torch.randn(n_pairs, feat_dim // 2)
doc_feats = torch.randn(n_pairs, feat_dim // 2)

# Cross feature: 쿼리와 문서를 결합
X = torch.cat([query_feats, doc_feats, query_feats * doc_feats[:, :feat_dim//2]], dim=1)  # (n, 24)
in_dim = X.shape[1]

# 관련도: 쿼리-문서 특성의 내적 기반 + 비선형
relevance_score = (query_feats * doc_feats).sum(dim=1)
y = (relevance_score > relevance_score.median()).float()

split = int(0.8 * n_pairs)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"pairs: {n_pairs}, features: {in_dim}")
print(f"train: {split}, test: {n_pairs - split}")

# --- Cross-Encoder ---
class CrossEncoder(nn.Module):
    """쿼리와 문서를 결합하여 관련도 점수를 직접 예측"""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = CrossEncoder(in_dim)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- Pairwise Loss (LambdaRank 유사) ---
def pairwise_loss(scores, labels):
    """관련 문서가 비관련 문서보다 높은 점수를 갖도록"""
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0)

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    # 모든 pos-neg 쌍에 대해 margin loss
    n_pos = min(pos_scores.size(0), 50)
    n_neg = min(neg_scores.size(0), 50)
    pos_sample = pos_scores[:n_pos].unsqueeze(1)  # (n_pos, 1)
    neg_sample = neg_scores[:n_neg].unsqueeze(0)  # (1, n_neg)

    # Sigmoid cross-entropy on score differences
    diff = pos_sample - neg_sample  # (n_pos, n_neg)
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss

# --- 학습 ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 256

for epoch in range(50):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        scores = model(X_train[idx])
        loss = pairwise_loss(scores, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_scores = model(X_test)
            test_acc = ((test_scores > 0).float() == y_test).float().mean().item() * 100
            # NDCG@5 계산
            sorted_idx = test_scores.argsort(descending=True)[:10]
            dcg = sum(y_test[idx].item() / torch.log2(torch.tensor(rank + 2.0)).item()
                      for rank, idx in enumerate(sorted_idx))
            ideal_sorted = y_test.argsort(descending=True)[:10]
            idcg = sum(y_test[idx].item() / torch.log2(torch.tensor(rank + 2.0)).item()
                       for rank, idx in enumerate(ideal_sorted))
            ndcg = dcg / (idcg + 1e-8)
        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | acc {test_acc:.1f}% | NDCG@10 {ndcg:.3f}")
