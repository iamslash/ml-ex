"""
Semantic Search with PyTorch.
문장 임베딩 기반 의미적 검색.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# --- 간단한 문장 임베딩 모델 ---
# 실제로는 Sentence-BERT를 사용하지만, 학습 목적으로 직접 구현

vocab = {}  # word -> id
def build_vocab(sentences):
    vocab['<pad>'] = 0
    idx = 1
    for sent in sentences:
        for word in sent.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return len(vocab)

def encode_sentence(sent, max_len=10):
    ids = [vocab.get(w, 0) for w in sent.lower().split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# --- 문서와 쿼리 ---
documents = [
    "the cat sat on the mat",
    "a dog played in the park",
    "machine learning is artificial intelligence",
    "deep learning uses neural networks",
    "natural language processing with transformers",
    "recommendation systems predict preferences",
    "search engines use inverted indexes",
    "reinforcement learning trains agents",
    "computer vision recognizes images",
    "graph neural networks model relationships",
]

queries_and_relevant = [
    ("cat sitting on mat", 0),
    ("dog playing outside", 1),
    ("AI and ML models", 2),
    ("neural net deep learning", 3),
    ("NLP transformer models", 4),
]

vocab_size = build_vocab(documents + [q for q, _ in queries_and_relevant])
print(f"vocab: {vocab_size}, documents: {len(documents)}")

# --- 모델: 간단한 문장 인코더 ---
class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        mask = (x != 0).unsqueeze(-1).float()
        # Mean pooling (padding 제외)
        pooled = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return self.fc(pooled)  # (batch, embed_dim)

model = SentenceEncoder(vocab_size)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- Contrastive 학습 ---
# 같은 문서를 다르게 표현한 쿼리-문서 쌍으로 학습
train_pairs = [
    ("cat on mat", "the cat sat on the mat"),
    ("dog in park", "a dog played in the park"),
    ("ML AI", "machine learning is artificial intelligence"),
    ("deep neural nets", "deep learning uses neural networks"),
    ("NLP transformers", "natural language processing with transformers"),
    ("recommender systems", "recommendation systems predict preferences"),
    ("search index", "search engines use inverted indexes"),
    ("RL agents", "reinforcement learning trains agents"),
    ("image recognition", "computer vision recognizes images"),
    ("graph networks", "graph neural networks model relationships"),
]

optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    q_ids = torch.tensor([encode_sentence(q) for q, _ in train_pairs])
    d_ids = torch.tensor([encode_sentence(d) for _, d in train_pairs])

    q_emb = model(q_ids)
    d_emb = model(d_ids)

    # L2 normalize
    q_emb = q_emb / (q_emb.norm(dim=1, keepdim=True) + 1e-8)
    d_emb = d_emb / (d_emb.norm(dim=1, keepdim=True) + 1e-8)

    # In-batch contrastive loss
    scores = q_emb @ d_emb.T  # (10, 10)
    labels = torch.arange(len(train_pairs))
    loss = nn.CrossEntropyLoss()(scores * 20, labels)  # temperature scaling

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f}")

# --- 검색 ---
print("\n--- semantic search ---")
model.eval()
with torch.no_grad():
    doc_ids = torch.tensor([encode_sentence(d) for d in documents])
    doc_emb = model(doc_ids)
    doc_emb = doc_emb / (doc_emb.norm(dim=1, keepdim=True) + 1e-8)

    for query, expected_idx in queries_and_relevant:
        q_ids = torch.tensor([encode_sentence(query)])
        q_emb = model(q_ids)
        q_emb = q_emb / (q_emb.norm(dim=1, keepdim=True) + 1e-8)

        scores = (q_emb @ doc_emb.T).squeeze()
        top3 = scores.topk(3)

        hit = expected_idx in top3.indices.tolist()
        print(f"\nquery: \"{query}\" {'[HIT]' if hit else '[MISS]'}")
        for rank, (idx, score) in enumerate(zip(top3.indices.tolist(), top3.values.tolist()), 1):
            marker = " <-" if idx == expected_idx else ""
            print(f"  {rank}. [{score:.3f}] {documents[idx]}{marker}")
