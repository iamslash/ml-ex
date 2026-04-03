"""
Matrix Factorization from scratch.
순수 numpy로 구현한 협업 필터링 추천 시스템.
"""

import numpy as np

np.random.seed(42)

# --- 합성 데이터: 유저-아이템 평점 행렬 ---
n_users = 50
n_items = 30
n_ratings = 500

# 실제 잠재 요인 (ground truth)
k_true = 3
U_true = np.random.randn(n_users, k_true)
V_true = np.random.randn(n_items, k_true)
R_full = U_true @ V_true.T + 0.5 * np.random.randn(n_users, n_items)
R_full = np.clip(R_full, 1, 5)  # 1-5 범위

# 관측된 평점만 샘플링
indices = np.random.choice(n_users * n_items, n_ratings, replace=False)
rows, cols = np.unravel_index(indices, (n_users, n_items))
ratings = R_full[rows, cols]

# train/test split
split = int(0.8 * n_ratings)
train_rows, test_rows = rows[:split], rows[split:]
train_cols, test_cols = cols[:split], cols[split:]
train_ratings, test_ratings = ratings[:split], ratings[split:]

print(f"users: {n_users}, items: {n_items}")
print(f"train: {len(train_ratings)}, test: {len(test_ratings)}")

# --- Matrix Factorization: R ≈ U @ V^T ---
k = 5  # 잠재 차원
U = np.random.randn(n_users, k) * 0.1
V = np.random.randn(n_items, k) * 0.1
bu = np.zeros(n_users)  # 유저 bias
bi = np.zeros(n_items)  # 아이템 bias
mu = train_ratings.mean()  # 글로벌 평균

# --- SGD 학습 ---
learning_rate = 0.01
reg = 0.02  # L2 정규화
num_epochs = 50

for epoch in range(num_epochs):
    # 학습 데이터 셔플
    perm = np.random.permutation(len(train_ratings))

    total_loss = 0
    for idx in perm:
        u, i, r = train_rows[idx], train_cols[idx], train_ratings[idx]

        # 예측: mu + bu + bi + U_u · V_i
        pred = mu + bu[u] + bi[i] + U[u] @ V[i]
        error = r - pred
        total_loss += error ** 2

        # 그래디언트 업데이트
        bu[u] += learning_rate * (error - reg * bu[u])
        bi[i] += learning_rate * (error - reg * bi[i])
        U[u] += learning_rate * (error * V[i] - reg * U[u])
        V[i] += learning_rate * (error * U[u] - reg * V[i])

    train_rmse = np.sqrt(total_loss / len(train_ratings))

    # 테스트 RMSE
    test_preds = mu + bu[test_rows] + bi[test_cols] + np.sum(U[test_rows] * V[test_cols], axis=1)
    test_rmse = np.sqrt(np.mean((test_ratings - test_preds) ** 2))

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:2d} | train RMSE {train_rmse:.4f} | test RMSE {test_rmse:.4f}")

# --- 추천 생성 ---
print("\n--- recommendations ---")
R_pred = mu + bu[:, None] + bi[None, :] + U @ V.T

for user_id in [0, 5, 10]:
    # 이미 평가한 아이템 제외
    rated = set(train_cols[train_rows == user_id])
    scores = R_pred[user_id]
    unrated = [(i, scores[i]) for i in range(n_items) if i not in rated]
    top5 = sorted(unrated, key=lambda x: x[1], reverse=True)[:5]
    items = [f"item{i}({s:.1f})" for i, s in top5]
    print(f"  user {user_id}: {', '.join(items)}")

# --- 임베딩 유사도 ---
print("\n--- similar items (by embedding) ---")
from numpy.linalg import norm

def cosine_sim(a, b):
    return a @ b / (norm(a) * norm(b) + 1e-8)

for item_id in [0, 5]:
    sims = [(j, cosine_sim(V[item_id], V[j])) for j in range(n_items) if j != item_id]
    top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
    items = [f"item{j}({s:.2f})" for j, s in top3]
    print(f"  item {item_id} similar: {', '.join(items)}")
