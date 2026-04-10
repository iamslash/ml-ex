# 3축 메모리 검색 (Generative Agents Memory Retrieval)

> 최신성(Recency) + 중요도(Importance) + 관련성(Relevance) 세 축의 가중 합산으로 에이전트 메모리를 검색하는 방식

## 실행 방법

```bash
cd agent-ex/3-memory-retrieval
python memory_retrieval.py
```

의존성: Python 표준 라이브러리 + numpy

## 핵심 개념

### 3축 점수

| 축 | 함수 | 설명 |
|---|---|---|
| 최신성 (Recency) | `recency_score` | 시간 경과에 따른 지수 감쇠 (`0.995^시간`) |
| 중요도 (Importance) | `importance_score` | 1~10 점수를 [0,1]로 정규화 |
| 관련성 (Relevance) | `relevance_score` | 쿼리와 메모리 임베딩의 코사인 유사도 |

### 최종 점수 공식

```
score = α·recency + β·importance + γ·relevance
```

α, β, γ 는 검색 목적에 따라 조절 가능한 가중치다.

### 임베딩 방식

외부 모델 없이 **Bag-of-Words** 벡터를 사용한다.

```python
def embed(text: str) -> np.ndarray:
    # 전체 텍스트로 구성된 고정 어휘에서 단어 빈도 벡터 생성
```

실제 시스템에서는 sentence-transformers 등의 밀집 임베딩으로 교체하면 된다.

### 하이브리드 vs 관련성 전용

```
관련성 전용 top-3:
  1. database connection timeout error occurred in production   ← 최신 + 고관련
  2. database schema migration completed successfully           ← 오래됨, 부분 관련
  3. connection pool exhausted under heavy load                 ← 5일 전, 부분 관련

하이브리드 top-3:
  1. database connection timeout error occurred in production   ← 최신 + 고관련 + 중요도 9
  2. fixed bug in user authentication login flow                ← 최신 + 중요도 7
  3. database schema migration completed successfully
```

하이브리드는 **최근에 발생한 중요한 사건**을 더 적극적으로 부각시킨다.

### MemoryRetriever API

```python
retriever = MemoryRetriever()
retriever.add_memory(entry)
results = retriever.retrieve_top_k(
    query="database connection timeout error",
    k=5,
    alpha=1.0,   # 최신성 가중치
    beta=1.0,    # 중요도 가중치
    gamma=1.0,   # 관련성 가중치
)
```

## 참고 논문/자료

- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023) https://arxiv.org/abs/2304.03442
- Zep AI Blog, "A Survey on the Memory of AI Agents" (2024) https://www.getzep.com/ai-agent-memory
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) — 검색 관련성 스코어링 기초
