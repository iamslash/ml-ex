"""
Self-Attention from scratch.
순수 numpy로 구현한 스케일드 닷 프로덕트 어텐션.
"""

import numpy as np

np.random.seed(42)

# --- 설정 ---
seq_len = 4       # 시퀀스 길이 (단어 수)
d_model = 8       # 임베딩 차원
d_k = 4           # Q, K 차원
d_v = 4           # V 차원

# --- 입력: 4개 단어의 임베딩 벡터 ---
# "I love ML models" 같은 문장을 상상
words = ["I", "love", "ML", "models"]
X = np.random.randn(seq_len, d_model)  # (4, 8)

print(f"input shape: {X.shape}  (seq_len={seq_len}, d_model={d_model})")
print(f"Q,K dim: {d_k}, V dim: {d_v}")

# --- Q, K, V 가중치 행렬 ---
W_Q = np.random.randn(d_model, d_k) * 0.3  # (8, 4)
W_K = np.random.randn(d_model, d_k) * 0.3  # (8, 4)
W_V = np.random.randn(d_model, d_v) * 0.3  # (8, 4)

# --- Step 1: Q, K, V 계산 ---
Q = X @ W_Q  # (4, 4) Query: "나는 무엇을 찾고 있는가"
K = X @ W_K  # (4, 4) Key: "나는 무엇을 갖고 있는가"
V = X @ W_V  # (4, 4) Value: "나의 실제 정보"

print(f"\nQ shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# --- Step 2: Attention Score ---
# score(i, j) = Q_i · K_j / sqrt(d_k)
scores = Q @ K.T / np.sqrt(d_k)  # (4, 4)

print(f"\nattention scores (before softmax):")
print(f"  shape: {scores.shape}")
for i, word in enumerate(words):
    print(f"  {word:>6s} -> {scores[i].round(3)}")

# --- Step 3: Softmax ---
def softmax(x):
    x_shift = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

weights = softmax(scores)  # (4, 4)

print(f"\nattention weights (after softmax):")
print(f"  각 행의 합 = 1 (확률 분포)")
header = "".join(f"{w:>8s}" for w in words)
print(f"  {'':>6s}{header}")
for i, word in enumerate(words):
    row = "".join(f"{weights[i,j]:8.3f}" for j in range(seq_len))
    print(f"  {word:>6s}{row}")

# --- Step 4: Weighted sum of Values ---
output = weights @ V  # (4, 4)

print(f"\noutput shape: {output.shape}")
print(f"각 단어의 출력은 모든 단어의 Value의 가중 합이다.")

# --- Causal (Masked) Attention ---
print("\n" + "="*50)
print("Causal Attention (GPT style)")
print("="*50)

# 미래 토큰을 볼 수 없도록 마스킹
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)  # upper triangle = -inf
masked_scores = scores + mask

causal_weights = softmax(masked_scores)

print(f"\ncausal attention weights:")
print(f"  각 단어는 자기 자신과 이전 단어만 볼 수 있다.")
header = "".join(f"{w:>8s}" for w in words)
print(f"  {'':>6s}{header}")
for i, word in enumerate(words):
    row = "".join(f"{causal_weights[i,j]:8.3f}" for j in range(seq_len))
    print(f"  {word:>6s}{row}")

causal_output = causal_weights @ V

# --- Multi-Head Attention ---
print("\n" + "="*50)
print("Multi-Head Attention")
print("="*50)

n_heads = 2
head_dim = d_model // n_heads  # 4

# 각 헤드별 Q, K, V
W_Qs = [np.random.randn(d_model, head_dim) * 0.3 for _ in range(n_heads)]
W_Ks = [np.random.randn(d_model, head_dim) * 0.3 for _ in range(n_heads)]
W_Vs = [np.random.randn(d_model, head_dim) * 0.3 for _ in range(n_heads)]
W_O = np.random.randn(n_heads * head_dim, d_model) * 0.3  # 출력 프로젝션

head_outputs = []
for h in range(n_heads):
    Q_h = X @ W_Qs[h]  # (4, 4)
    K_h = X @ W_Ks[h]
    V_h = X @ W_Vs[h]

    scores_h = Q_h @ K_h.T / np.sqrt(head_dim)
    weights_h = softmax(scores_h)
    output_h = weights_h @ V_h
    head_outputs.append(output_h)

    print(f"\nhead {h} attention weights:")
    for i, word in enumerate(words):
        row = "".join(f"{weights_h[i,j]:8.3f}" for j in range(seq_len))
        print(f"  {word:>6s}{row}")

# Concatenate + Output projection
concat = np.concatenate(head_outputs, axis=-1)  # (4, 8)
multi_head_output = concat @ W_O  # (4, 8)

print(f"\nmulti-head output shape: {multi_head_output.shape}")
print(f"각 헤드는 다른 관계 패턴에 집중한다.")
