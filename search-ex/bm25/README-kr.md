# BM25 (Best Match 25)

> 순수 Python으로 구현한 고전 정보 검색 알고리즘. TF-IDF를 개선한 확률적 랭킹 모델.

## 실행 방법

```bash
cd search-ex/bm25
python bm25.py
```

의존성 없음. 표준 라이브러리(`math`, `collections`)만 사용.

**출력 예시:**
```
query: "cat and dog"
  1. [1.234] a cat and a dog are friends
  2. [0.876] the cat sat on the mat
  3. [0.543] the dog played in the park
```

## 핵심 개념

### BM25 공식

```
BM25(q, d) = Σ IDF(t) * TF_norm(t, d)
```

**IDF (Inverse Document Frequency):**
```
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```
- `N`: 전체 문서 수
- `df(t)`: 단어 `t`가 등장하는 문서 수
- 희귀한 단어일수록 높은 IDF 점수

**TF 정규화 (Term Frequency Normalization):**
```
TF_norm(t, d) = tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```
- `tf(t,d)`: 문서 d에서 단어 t의 빈도
- `k1=1.5`: TF 포화 파라미터 (값이 클수록 TF의 영향 증가)
- `b=0.75`: 문서 길이 정규화 강도 (0=정규화 없음, 1=완전 정규화)
- `|d|`: 문서 길이, `avgdl`: 평균 문서 길이

### 역색인 (Inverted Index)

각 단어가 어떤 문서에 등장하는지를 미리 계산해두는 자료구조. `df` (document frequency) 딕셔너리로 구현.

### 핵심 파라미터 역할

| 파라미터 | 기본값 | 역할 |
|---------|--------|------|
| `k1` | 1.5 | TF 포화점 조절. 높을수록 고빈도 단어 우대 |
| `b` | 0.75 | 문서 길이 편향 보정. 1.0이면 완전 보정 |

## 관련 모델과의 비교

| 모델 | 방식 | 장점 | 단점 |
|------|------|------|------|
| **BM25** | 키워드 매칭 + 통계 | 빠름, 해석 가능, 추가 학습 불필요 | 동의어/의미 파악 불가 |
| **TF-IDF** | 키워드 매칭 | 단순, 빠름 | 문서 길이 편향 있음 |
| **Semantic Search** | 임베딩 유사도 | 의미적 유사성 파악 | 학습 필요, 느림 |
| **BM25 + Reranker** | 하이브리드 | 속도 + 정밀도 균형 | 2단계 파이프라인 복잡도 |

**실무 활용 패턴:**
- BM25는 대규모 문서에서 1차 후보 검색(recall)에 주로 사용
- 이후 Cross-Encoder Reranker로 정밀 정렬(precision) 수행
- Elasticsearch, OpenSearch 등 검색 엔진의 기본 랭킹 알고리즘으로 탑재됨
