# Neural Collaborative Filtering (신경 협업 필터링)

> GMF(일반화 행렬 분해)와 MLP를 결합하여 유저-아이템 상호작용의 선형/비선형 패턴을 동시에 학습하는 모델

## 실행 방법

```bash
cd recsys-ex/ncf
python ncf.py
```

의존성: `torch`

출력 예시:
- 에폭별 train/test loss 및 정확도 (5 에폭마다)
- 유저별 top-5 추천 아이템 (sigmoid 확률)

## 핵심 개념

### 아키텍처

NCF는 두 경로(GMF + MLP)를 병렬로 실행하고 결합합니다:

```
User ID          Item ID
   |                 |
GMF_user_emb    GMF_item_emb      MLP_user_emb    MLP_item_emb
   |                 |                  |                |
   └──── 요소별 곱 ────┘          └──── concat ────┘
          (B, 16)                      (B, 32)
             |                            |
          GMF out                    MLP layers
                                  [32->32, 32->16]
             |                            |
             └─────── concat ─────────────┘
                        (B, 32)
                           |
                       fc_out -> 클릭 확률
```

### GMF (Generalized Matrix Factorization)

```python
gmf_out = gmf_user_emb(u) * gmf_item_emb(i)  # 요소별 곱
```

- 전통적 MF의 내적을 일반화
- 선형 관계를 포착

### MLP 경로

```python
mlp_input = cat([mlp_user_emb(u), mlp_item_emb(i)])  # concat
mlp_out = MLP(mlp_input)  # [Linear+ReLU 스택]
```

- concat 후 비선형 변환
- 복잡한 상호작용 패턴 포착

### Implicit Feedback

```python
# Positive: 실제 상호작용
pos_labels = ones(n_pos)
# Negative: 무작위 샘플링 (상호작용 없는 쌍)
neg_labels = zeros(n_neg)
```

- 명시적 평점 대신 클릭/구매 여부 (0/1)를 학습
- BCEWithLogitsLoss로 이진 분류

### 분리된 임베딩
GMF와 MLP 경로는 별도의 임베딩 테이블을 사용:
- GMF: `gmf_user_emb`, `gmf_item_emb`
- MLP: `mlp_user_emb`, `mlp_item_emb`

같은 임베딩을 공유하면 두 경로의 학습 목표가 충돌할 수 있음

## 관련 모델과의 비교

| 모델 | 입력 | 비선형성 | Implicit Feedback | 특징 |
|------|------|----------|-------------------|------|
| **NCF** | ID | GMF + MLP | 지원 | 두 경로 병렬 |
| **Matrix Factorization** | ID | 없음 (내적) | 부분 지원 | 단순, 해석 쉬움 |
| **Two-Tower** | 피처 | MLP | 지원 | 대규모 검색 |
| **DCN-v2** | 피처 | Cross + MLP | 지원 | 고차 교차항 |

### NCF vs MF
- MF는 내적 (선형)만 사용 → 단순한 협업 신호 포착
- NCF-GMF는 MF의 일반화 버전
- NCF-MLP는 비선형 패턴 추가 학습
- NCF (GMF+MLP)는 두 강점을 결합

### NCF vs Two-Tower
- NCF: 유저/아이템 ID만 입력, 사이드 피처 미사용
- Two-Tower: 피처 벡터를 입력으로 사용하여 cold-start 대응
- NCF는 랭킹, Two-Tower는 검색 단계에 더 적합

### NCF vs FT-Transformer
- NCF: ID 기반, 협업 필터링 특화
- FT-Transformer: 수치형/범주형 피처를 토큰화하여 어텐션 학습
- 피처가 풍부한 환경에서는 FT-Transformer가 유리

### 참고 논문
- He et al., "Neural Collaborative Filtering" (WWW 2017)
