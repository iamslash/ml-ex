# Self-Attention (셀프 어텐션)

> Transformer의 핵심 메커니즘. 시퀀스 내 모든 위치 간의 관계를 학습한다.

## 실행 방법

```bash
cd scratch-ex/self-attention
python self_attention.py
```

## 핵심 개념

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

4단계로 구성:

| 단계 | 수식 | 의미 |
|------|------|------|
| 1. 프로젝션 | Q=XW_Q, K=XW_K, V=XW_V | 입력을 Query, Key, Value로 변환 |
| 2. 점수 | score = Q @ K^T / sqrt(d_k) | 각 단어 쌍의 관련도 계산 |
| 3. 가중치 | weights = softmax(score) | 점수를 확률로 변환 |
| 4. 출력 | output = weights @ V | Value의 가중 합 |

### Q, K, V 직관

- **Query**: "나는 무엇을 찾고 있는가" (질문)
- **Key**: "나는 무엇에 대한 정보인가" (인덱스)
- **Value**: "나의 실제 정보" (내용)

검색엔진과 유사: Query로 검색 → Key와 매칭 → 매칭된 Value를 반환

### Causal Attention (GPT 스타일)

미래 토큰을 볼 수 없도록 마스킹. 상삼각 행렬에 -inf를 넣어 softmax 후 0이 되게 한다.

```
"I"      → [I]만 참조
"love"   → [I, love] 참조
"ML"     → [I, love, ML] 참조
"models" → [I, love, ML, models] 참조
```

### Multi-Head Attention

여러 개의 어텐션을 병렬로 수행. 각 헤드가 다른 종류의 관계에 집중한다.

```
head_0: 문법적 관계에 집중 (주어-동사)
head_1: 의미적 관계에 집중 (ML-models)
```

결과를 연결(concat)한 뒤 출력 프로젝션으로 합친다.

### CNN과의 비교

| | CNN | Self-Attention |
|---|---|---|
| 수용 범위 | 로컬 (커널 크기) | 글로벌 (전체 시퀀스) |
| 위치 관계 | 인접한 것만 | 어디든 직접 연결 |
| 계산량 | O(n * k) | O(n²) |
| 용도 | 이미지 | 시퀀스 (텍스트, 음성) |

## microgpt와의 관계

microgpt의 `gpt()` 함수 안에 있는 어텐션 블록이 바로 이것이다:
- Q, K, V 프로젝션 → 스케일드 닷 프로덕트 → 가중합
- KV Cache를 사용한 causal attention
- Multi-Head로 병렬 처리
