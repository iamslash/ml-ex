# Item2Vec (아이템 임베딩)

> Word2Vec의 Skip-gram을 행동 시퀀스에 적용하여 아이템의 의미적 유사성을 학습하는 임베딩 모델

## 실행 방법

```bash
cd recsys-ex/item2vec
python item2vec.py
```

의존성: `numpy`만 필요

출력 예시:
- 에폭별 손실 (5 에폭마다)
- 아이템별 유사 아이템 (같은 클러스터 여부 표시)
- 유저 임베딩 기반 추천 결과

## 핵심 개념

### Skip-gram 학습 쌍 생성

```
시퀀스: [item_0, item_3, item_1, item_7, item_2]
윈도우 크기: 3

center=item_3 의 context: [item_0, item_1, item_7]
-> 쌍: (3, 0), (3, 1), (3, 7)
```

Word2Vec에서 단어를 아이템으로, 문장을 행동 시퀀스로 대체

### Negative Sampling

```python
# 빈도 기반 분포 (0.75승으로 smoothing)
neg_dist = freq_array ** 0.75
neg_dist /= neg_dist.sum()

# 학습: positive pair에 대해 n_neg=5개 negative 샘플
```

- 자주 등장하는 아이템이 너무 자주 negative로 선택되는 것을 방지
- 0.75승: 고빈도 아이템의 확률을 낮추고 저빈도 아이템의 확률을 높임

### 손실 함수

```
L = -log sigma(v_c · v_w)              # positive pair
  - sum log sigma(-v_c · v_neg)        # negative samples
```

- `v_c`: center 임베딩 (W_center)
- `v_w`: context 임베딩 (W_context)
- 두 임베딩 테이블을 분리하여 학습

### SGD 업데이트

```python
# Positive
grad_center = (prob - 1) * W_context[context]
W_center[center] -= lr * grad_center

# Negative
grad_center_neg = prob_neg * W_context[neg]
W_center[center] -= lr * grad_center_neg
```

### 최종 임베딩

```python
embeddings = (W_center + W_context) / 2
```

center + context 임베딩의 평균을 최종 표현으로 사용

### 유저 임베딩 (Cold-start 전략)

```python
# 전체 시퀀스 평균
u_emb = mean([embeddings[item] for item in seq])

# 최근 k개 평균
u_emb = mean([embeddings[item] for item in seq[-k:]])
```

별도의 유저 임베딩 없이 아이템 임베딩으로 유저 표현 생성

## 관련 모델과의 비교

| 모델 | 시퀀스 활용 | 피처 활용 | 학습 방식 | 출력 |
|------|------------|-----------|-----------|------|
| **Item2Vec** | Skip-gram | ID만 | 비지도 | 아이템 임베딩 |
| **Matrix Factorization** | 없음 | ID만 | 지도 (평점) | 유저+아이템 임베딩 |
| **Two-Tower** | 없음 | 피처 벡터 | 지도 (클릭) | 유저+아이템 임베딩 |
| **BERT4Rec** | Transformer | ID만 | 자기지도 | 시퀀스 표현 |
| **SASRec** | Self-Attention | ID만 | 자기지도 | 시퀀스 표현 |

### Item2Vec vs Word2Vec
- 완전히 동일한 알고리즘
- 단어 -> 아이템, 문장 -> 유저 행동 시퀀스
- Item2Vec은 순서 무관 (bag-of-items) 처리 가능

### Item2Vec vs Matrix Factorization
- MF: 유저-아이템 평점 행렬에서 ID 임베딩 학습
- Item2Vec: 행동 시퀀스의 공출현에서 아이템 임베딩 학습
- Item2Vec은 평점 없는 implicit feedback에 더 자연스럽게 적용

### Item2Vec vs BERT4Rec
- Item2Vec: 정적 임베딩, 순서 정보 제한적 활용
- BERT4Rec: Transformer로 순서와 맥락을 정밀하게 포착
- Item2Vec은 훨씬 가볍고 대규모 아이템 카탈로그에 효율적

### 응용
- 아이템 유사도 기반 연관 추천 (장바구니 분석)
- 검색 시스템의 아이템 임베딩 초기화
- Two-Tower 또는 DCN의 아이템 피처로 활용

### 참고 논문
- Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative Filtering" (MLSP 2016)
- Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013)
