# Matrix Factorization (행렬 분해)

> 유저-아이템 평점 행렬을 잠재 요인 벡터로 분해하여 협업 필터링을 수행하는 고전적 추천 모델

## 실행 방법

```bash
cd recsys-ex/matrix-factorization
python matrix_factorization.py
```

의존성: `numpy`만 필요 (PyTorch 불필요)

출력 예시:
- 에폭별 train/test RMSE
- 유저별 top-5 추천 아이템
- 아이템 임베딩 기반 유사 아이템

## 핵심 개념

### 모델 수식
평점 예측값:

```
r̂(u, i) = μ + b_u + b_i + U_u · V_i
```

| 기호 | 의미 |
|------|------|
| `μ` | 전체 평균 평점 (글로벌 바이어스) |
| `b_u` | 유저 바이어스 (특정 유저의 후한/박한 성향) |
| `b_i` | 아이템 바이어스 (인기도 등) |
| `U_u` | 유저 잠재 벡터 (k차원) |
| `V_i` | 아이템 잠재 벡터 (k차원) |

### SGD 업데이트
오차 `e = r - r̂`에 대해 L2 정규화 포함:

```
b_u += lr * (e - reg * b_u)
b_i += lr * (e - reg * b_i)
U_u += lr * (e * V_i - reg * U_u)
V_i += lr * (e * U_u - reg * V_i)
```

### 구현 특징
- **순수 NumPy 구현**: PyTorch 없이 SGD를 직접 구현
- **잠재 차원 k=5**: ground truth k=3보다 크게 설정해 오버피팅 방지
- **L2 정규화 (reg=0.02)**: 임베딩 과대학습 억제
- **코사인 유사도**: 학습된 V 행렬로 유사 아이템 검색

### 추천 생성
```python
R_pred = μ + bu[:, None] + bi[None, :] + U @ V.T
```
이미 평가한 아이템은 제외하고 점수 높은 순으로 정렬

## 관련 모델과의 비교

| 모델 | 방식 | 특성 교차 | 확장성 | 비고 |
|------|------|-----------|--------|------|
| **Matrix Factorization** | 잠재 요인 분해 | 내적(선형) | 높음 | 해석 용이, explicit feedback |
| **NCF** | 신경망 | GMF + MLP | 중간 | 비선형 패턴 포착 |
| **Two-Tower** | 듀얼 인코더 | 내적 | 매우 높음 | 피처 활용, 검색에 특화 |
| **Item2Vec** | 행동 시퀀스 | Skip-gram | 높음 | implicit feedback, 순서 정보 활용 |

### MF vs NCF
- MF는 내적만 사용하여 선형 관계만 포착
- NCF는 MLP를 추가하여 비선형 상호작용을 학습
- MF는 계산 효율이 높고 해석이 쉬움

### MF vs ALS (Alternating Least Squares)
- 이 구현은 SGD 기반 (sparse 데이터에 효율적)
- ALS는 각 파라미터를 해석적으로 풀어 병렬화가 쉬움 (Spark MLlib 등)

### 참고 논문
- Koren et al., "Matrix Factorization Techniques for Recommender Systems" (2009)
- He et al., "Neural Collaborative Filtering" (2017)
