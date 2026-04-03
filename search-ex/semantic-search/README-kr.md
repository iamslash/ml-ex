# Semantic Search (의미 기반 검색)

> 문장 임베딩과 대조 학습으로 의미적 유사성을 측정하는 뉴럴 검색 모델.

## 실행 방법

```bash
cd search-ex/semantic-search
pip install torch numpy
python semantic_search.py
```

**출력 예시:**
```
epoch  20 | loss 0.0312
epoch  40 | loss 0.0098
...
query: "cat sitting on mat" [HIT]
  1. [0.921] the cat sat on the mat  <-
  2. [0.743] a dog played in the park
  3. [0.612] graph neural networks model relationships
```

## 핵심 개념

### 문장 인코더 (SentenceEncoder)

단어 임베딩 → 평균 풀링 → 완전연결층으로 구성:

```
입력 문장 -> 토큰 ID 시퀀스
         -> Embedding(vocab_size, embed_dim=32)
         -> Mean Pooling (패딩 토큰 제외)
         -> FC(embed_dim -> hidden_dim=64 -> embed_dim)
         -> L2 정규화 -> 단위 벡터
```

패딩 마스크를 사용해 `<pad>` 토큰을 풀링에서 제외하여 정확한 평균을 계산.

### In-Batch 대조 학습 (Contrastive Learning)

쿼리-문서 양성 쌍 N개를 배치로 구성, 모든 쌍에 대한 유사도 행렬로 학습:

```python
scores = q_emb @ d_emb.T  # (N, N) 유사도 행렬
labels = torch.arange(N)  # 대각선이 양성 쌍
loss = CrossEntropyLoss(scores * 20, labels)  # temperature=20
```

- 같은 인덱스의 쿼리-문서 쌍은 양성(positive), 나머지는 음성(negative)
- Temperature scaling(`* 20`)으로 분포를 날카롭게 만들어 학습 안정화

### 코사인 유사도 검색

모든 문서를 사전에 임베딩하고 L2 정규화 후 내적으로 코사인 유사도 계산:

```python
similarity = q_emb @ doc_emb.T  # 정규화된 벡터의 내적 = 코사인 유사도
```

### Mean Pooling과 패딩 처리

```python
mask = (x != 0).unsqueeze(-1).float()
pooled = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
```

패딩 토큰(`id=0`)을 마스킹하여 실제 단어만으로 평균을 계산.

## 관련 모델과의 비교

| 모델 | 구조 | 검색 방식 | 의미 파악 |
|------|------|----------|----------|
| **Semantic Search (Bi-Encoder)** | 쿼리/문서 각각 인코딩 | ANN 인덱스 | 가능 |
| **BM25** | 역색인 | 정확 매칭 | 불가 |
| **Cross-Encoder (Reranker)** | 쿼리+문서 결합 인코딩 | 쌍별 스코어링 | 가능 |
| **Sentence-BERT** | BERT 기반 Bi-Encoder | ANN 인덱스 | 우수 |

**Bi-Encoder vs Cross-Encoder 트레이드오프:**
- Bi-Encoder(이 예제): 문서를 사전 임베딩 가능 → 빠른 검색, but 정밀도 낮음
- Cross-Encoder: 쿼리-문서를 쌍으로 처리 → 높은 정밀도, but 전체 코퍼스 실시간 처리 불가

**실무 파이프라인:**
```
BM25/Bi-Encoder (빠른 1차 검색, Top-K 후보) -> Cross-Encoder (정밀 재랭킹)
```
