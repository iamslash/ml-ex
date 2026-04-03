# Cross-Encoder Reranker (교차 인코더 재랭커)

> 1차 검색 결과를 쿼리-문서 쌍 단위로 정밀 점수화하는 재랭킹 모델. Pairwise Loss와 NDCG로 평가.

## 실행 방법

```bash
cd search-ex/reranker
pip install torch
python reranker.py
```

**출력 예시:**
```
pairs: 2000, features: 24
train: 1600, test: 400
epoch 10 | loss 0.3241 | acc 84.5% | NDCG@10 0.912
epoch 20 | loss 0.2876 | acc 87.2% | NDCG@10 0.934
...
```

## 핵심 개념

### Cross-Encoder 구조

쿼리와 문서 특성을 결합하여 하나의 관련도 점수를 출력:

```
입력: [query_feats | doc_feats | query * doc]  (24차원)
   -> Linear(24, 64) -> ReLU -> Dropout(0.1)
   -> Linear(64, 32) -> ReLU -> Dropout(0.1)
   -> Linear(32, 1)
   -> 스칼라 관련도 점수
```

Cross feature(`query_feats * doc_feats`)를 포함하여 쿼리-문서 상호작용을 명시적으로 모델링.

### Pairwise Loss (쌍별 손실)

관련 문서가 비관련 문서보다 높은 점수를 갖도록 학습:

```python
diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # (n_pos, n_neg)
loss = -log(sigmoid(diff))
```

- 모든 양성-음성 쌍의 조합에 대해 마진 손실 계산
- `sigmoid(diff) -> 1.0`이면 양성이 음성보다 확실히 높은 점수
- LambdaRank, LambdaMART와 유사한 접근법

### NDCG (Normalized Discounted Cumulative Gain)

랭킹 품질을 측정하는 지표:

```
DCG@K  = Σ(rank=1..K) relevance[rank] / log2(rank + 1)
NDCG@K = DCG@K / IDCG@K
```

- `IDCG`: 이상적인 정렬 순서의 DCG (상한값)
- NDCG = 1.0이면 완벽한 랭킹, 0.0이면 최악의 랭킹
- 상위 순위에 더 높은 가중치 부여 (log2 감쇠)

### Dropout을 통한 정규화

Reranker는 쿼리-문서 쌍이 한정적이므로 과적합 위험이 높음. Dropout(0.1)으로 일반화 향상.

## 관련 모델과의 비교

| 모델 | 손실 함수 | 입력 형태 | 속도 | 정밀도 |
|------|----------|----------|------|--------|
| **Cross-Encoder (이 예제)** | Pairwise Loss | 쿼리+문서 결합 | 느림 | 높음 |
| **Bi-Encoder** | Contrastive Loss | 쿼리, 문서 독립 | 빠름 | 중간 |
| **BM25** | 없음 (통계) | 키워드 | 매우 빠름 | 낮음 |
| **LambdaMART** | LambdaRank | 특성 벡터 | 중간 | 높음 |

**Pointwise vs Pairwise vs Listwise:**

| 방식 | 학습 단위 | 손실 예시 | 특징 |
|------|----------|----------|------|
| Pointwise | 문서 1개 | MSE, BCE | 단순, 상대적 순서 무시 |
| Pairwise | 문서 쌍 | BPR, Hinge | 상대 순서 학습, 이 예제 방식 |
| Listwise | 문서 리스트 전체 | ListNet, SoftmaxNDCG | 전체 랭킹 최적화 |

**실무 위치:**
```
질의 -> BM25/Bi-Encoder (Top-100 후보) -> Cross-Encoder Reranker (Top-10 정밀 정렬) -> 사용자
```
