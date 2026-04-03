"""
Item2Vec from scratch.
행동 시퀀스로부터 아이템 임베딩을 학습하는 Word2Vec 응용.
"""

import numpy as np
from collections import Counter

np.random.seed(42)

# --- 합성 데이터: 유저 행동 시퀀스 ---
n_items = 50
n_users = 200
embed_dim = 16

# 아이템 클러스터 (유사한 아이템끼리 함께 소비되는 경향)
clusters = {
    0: list(range(0, 10)),    # 액션 영화
    1: list(range(10, 20)),   # 로맨스 영화
    2: list(range(20, 30)),   # SF 영화
    3: list(range(30, 40)),   # 코미디 영화
    4: list(range(40, 50)),   # 다큐멘터리
}

def generate_sequence(length=10):
    """유저 행동 시퀀스 생성: 같은 클러스터에서 자주 선택"""
    preferred = np.random.choice(list(clusters.keys()), p=[0.3, 0.25, 0.2, 0.15, 0.1])
    seq = []
    for _ in range(length):
        if np.random.rand() < 0.7:  # 70% 확률로 선호 클러스터
            item = np.random.choice(clusters[preferred])
        else:
            item = np.random.randint(0, n_items)
        seq.append(item)
    return seq

sequences = [generate_sequence(np.random.randint(5, 15)) for _ in range(n_users)]
print(f"items: {n_items}, users: {n_users}, sequences: {len(sequences)}")
print(f"avg sequence length: {np.mean([len(s) for s in sequences]):.1f}")

# --- Skip-gram 학습 쌍 생성 ---
window_size = 3

pairs = []  # (center, context)
for seq in sequences:
    for i, center in enumerate(seq):
        for j in range(max(0, i - window_size), min(len(seq), i + window_size + 1)):
            if i != j:
                pairs.append((center, seq[j]))

print(f"training pairs: {len(pairs)}")

# --- Negative Sampling ---
# 아이템 빈도 기반 negative 분포
item_freq = Counter()
for seq in sequences:
    item_freq.update(seq)

freq_array = np.array([item_freq.get(i, 1) for i in range(n_items)], dtype=np.float64)
neg_dist = freq_array ** 0.75  # 빈도 0.75승으로 smoothing
neg_dist /= neg_dist.sum()

# --- 임베딩 초기화 ---
W_center = np.random.randn(n_items, embed_dim) * 0.1  # center embeddings
W_context = np.random.randn(n_items, embed_dim) * 0.1  # context embeddings

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

# --- SGD 학습 ---
learning_rate = 0.025
n_neg_samples = 5
num_epochs = 20

for epoch in range(num_epochs):
    np.random.shuffle(pairs)
    total_loss = 0

    for center, context in pairs:
        # Positive: maximize log sigmoid(v_c · v_w)
        score = W_center[center] @ W_context[context]
        prob = sigmoid(score)
        loss = -np.log(prob + 1e-8)

        # Gradient for positive
        grad_center = (prob - 1) * W_context[context]
        grad_context = (prob - 1) * W_center[center]
        W_center[center] -= learning_rate * grad_center
        W_context[context] -= learning_rate * grad_context

        # Negative samples
        neg_items = np.random.choice(n_items, n_neg_samples, p=neg_dist)
        for neg in neg_items:
            score_neg = W_center[center] @ W_context[neg]
            prob_neg = sigmoid(score_neg)
            loss += -np.log(1 - prob_neg + 1e-8)

            grad_center_neg = prob_neg * W_context[neg]
            grad_context_neg = prob_neg * W_center[center]
            W_center[center] -= learning_rate * grad_center_neg
            W_context[neg] -= learning_rate * grad_context_neg

        total_loss += loss

    avg_loss = total_loss / len(pairs)
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch+1:2d} | loss {avg_loss:.4f}")

# --- 유사 아이템 검색 ---
print("\n--- similar items ---")

def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# 최종 임베딩: center + context 평균
embeddings = (W_center + W_context) / 2

for item_id in [0, 10, 20, 30, 40]:
    sims = [(j, cosine_similarity(embeddings[item_id], embeddings[j]))
            for j in range(n_items) if j != item_id]
    top5 = sorted(sims, key=lambda x: x[1], reverse=True)[:5]
    cluster_id = item_id // 10
    items = [f"{j}({'same' if j//10 == cluster_id else 'diff'})" for j, _ in top5]
    print(f"  item {item_id} (cluster {cluster_id}): {', '.join(items)}")

# --- 유저 임베딩 (cold-start 전략) ---
print("\n--- user embeddings ---")

def user_embedding_avg(seq, embeddings):
    """유저 시퀀스의 아이템 임베딩 평균"""
    return np.mean([embeddings[item] for item in seq], axis=0)

def user_embedding_recent(seq, embeddings, k=3):
    """최근 k개 아이템 임베딩 평균"""
    recent = seq[-k:]
    return np.mean([embeddings[item] for item in recent], axis=0)

for uid in [0, 50, 100]:
    seq = sequences[uid]
    u_emb = user_embedding_avg(seq, embeddings)
    # 유저 임베딩으로 추천
    scores = [(i, cosine_similarity(u_emb, embeddings[i])) for i in range(n_items)]
    top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    print(f"  user {uid} (seq: {seq[:5]}...) -> top items: {[i for i, _ in top3]}")
