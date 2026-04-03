# NCF (Neural Collaborative Filtering)

> GMF와 MLP를 결합한 신경망 기반 협업 필터링 랭킹 모델.

## 실행 방법

```bash
cd recsys-ex/ncf
python ncf.py
```

## 핵심 개념

### 구조

```
유저 ID → GMF 임베딩 ─┐
                       ├→ element-wise product → ┐
아이템 ID → GMF 임베딩 ─┘                         ├→ concat → FC → 확률
유저 ID → MLP 임베딩 ─┐                          │
                       ├→ concat → MLP → ────────┘
아이템 ID → MLP 임베딩 ─┘
```

- **GMF**: 선형 상호작용 (MF의 신경망 일반화)
- **MLP**: 비선형 상호작용 학습
- 두 경로를 결합하여 표현력 극대화

### Implicit Feedback

명시적 평점(1-5) 대신 클릭/구매 여부(0/1)로 학습. Negative sampling으로 미관측 상호작용을 부정 예시로 활용.
