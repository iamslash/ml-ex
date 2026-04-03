# Two-Tower Retrieval Model (투 타워 검색 모델)

> 유저와 아이템을 각각 독립된 신경망으로 인코딩하여 내적 기반 대규모 검색을 수행하는 모델

## 실행 방법

```bash
cd recsys-ex/two-tower
python two_tower.py
```

의존성: `torch`

출력 예시:
- 에폭별 in-batch negative 손실
- 유저별 top-5 추천 아이템 (코사인 유사도 기반)

## 핵심 개념

### 아키텍처

```
User Features (8차원)                Item Features (6차원)
       |                                    |
  UserTower                            ItemTower
  Linear(8->32) + ReLU                Linear(6->32) + ReLU
  Linear(32->16)                      Linear(32->16)
       |                                    |
  L2 Normalize                         L2 Normalize
       |                                    |
  user_emb (16차원)               item_emb (16차원)
              \                       /
               dot product (내적)
               → 유사도 점수
```

### In-batch Negative Sampling
배치 크기 B에서 B개의 positive 쌍이 있을 때, 배치 내 다른 아이템을 자동으로 negative로 활용:

```python
scores = user_emb @ item_emb.T  # (B, B) 행렬
labels = torch.arange(B)        # 대각선이 positive
loss = CrossEntropyLoss(scores, labels)
```

- 장점: 별도 negative 샘플링 없이 B-1개의 negative를 자동 생성
- 단점: 배치 내 우연한 positive가 false negative가 될 수 있음

### L2 정규화
두 타워 모두 출력을 단위 벡터로 정규화:

```python
emb = self.net(x)
return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
```

정규화 후 내적 = 코사인 유사도 → 점수 범위가 [-1, 1]로 안정화

### 서빙 전략

```
학습 시: user_emb · item_emb (실시간 계산)
서빙 시:
  - 아이템 임베딩을 사전에 계산하여 인덱스 구축 (FAISS 등)
  - 유저 임베딩만 실시간으로 계산
  - ANN (Approximate Nearest Neighbor) 검색으로 top-K 추출
```

## 관련 모델과의 비교

| 모델 | 사용 단계 | 지연시간 | 피처 활용 | 상호작용 표현력 |
|------|-----------|----------|-----------|----------------|
| **Two-Tower** | 검색(Retrieval) | 매우 낮음 | 풍부 | 낮음 (내적만) |
| **Matrix Factorization** | 검색/랭킹 | 낮음 | ID만 | 낮음 (내적만) |
| **DCN-v2** | 랭킹(Ranking) | 중간 | 풍부 | 높음 (교차항) |
| **NCF** | 랭킹 | 중간 | ID만 | 중간 (MLP) |

### Two-Tower vs Cross-Encoder
- Two-Tower: 유저/아이템 독립 인코딩 → ANN 검색 가능 → 수억 아이템 처리
- Cross-Encoder: 유저-아이템 쌍 입력 → 정확하지만 느림 → 소규모 재랭킹에 적합

### Two-Tower vs Matrix Factorization
- MF: ID 임베딩만 학습, 사이드 피처 활용 불가
- Two-Tower: 유저/아이템 피처를 신경망으로 인코딩, cold-start 대응 가능

### 실제 사용 사례
- YouTube DNN (2016): 검색 단계에 Two-Tower 구조 사용
- Google Play (2019): 앱 추천 검색에 적용
- Pinterest, Twitter 검색 시스템

### 참고 논문
- Covington et al., "Deep Neural Networks for YouTube Recommendations" (2016)
- Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (2019)
