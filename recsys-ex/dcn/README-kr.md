# Deep & Cross Network v2 (딥 앤 크로스 네트워크 v2)

> Cross Layer로 고차 피처 교차항을 자동으로 학습하고, Deep Network로 비선형 패턴을 포착하는 CTR 예측 모델

## 실행 방법

```bash
cd recsys-ex/dcn
python dcn.py
```

의존성: `torch`

출력 예시:
- 에폭별 train/test loss 및 정확도 (10 에폭마다)
- Cross Layer별 가중치 행렬 norm

## 핵심 개념

### 아키텍처

```
Input x (16차원)
       |
  +----+-----+
  |          |
Cross Path  Deep Path
  |          |
CrossLayer  Linear(16->64) + ReLU
x0*(W@xl+b) Linear(64->32) + ReLU
  + xl       |
  |          |
CrossLayer  deep_out (32차원)
  |          |
CrossLayer   |
  |          |
xl (16차원) |
  |          |
  +----+-----+
       |
  concat (16+32=48차원)
       |
  fc_out -> CTR 예측
```

### Cross Layer (핵심)

```python
# x_{l+1} = x_0 * (W @ x_l + b) + x_l
def forward(self, x0, xl):
    return x0 * self.W(xl) + xl
```

- `x_0`: 원본 입력을 매 레이어마다 재주입 (residual connection과 유사)
- `W @ xl`: 현재 레이어의 선형 변환
- `x_0 * (...)`: 요소별 곱으로 교차항 생성
- `+ xl`: skip connection으로 기울기 소실 방지

레이어 수에 따른 최고 교차 차수:

| 레이어 수 | 최고 교차 차수 |
|-----------|---------------|
| 1 | 2차 |
| 2 | 4차 |
| 3 | 8차 |

### 합성 데이터의 정답

```python
y = (X[:,0]*X[:,1] + X[:,2]*X[:,3] + sin(X[:,4]) > 0)
```

2차 교차항(`x_0 * x_1`, `x_2 * x_3`)을 포함하여 Cross Layer의 효과를 검증

### DCN-v1 vs DCN-v2 차이
- DCN-v1: `x_{l+1} = x_0 * x_l^T * w + b + x_l` (벡터 가중치, O(d) 파라미터)
- DCN-v2: `x_{l+1} = x_0 * (W @ x_l + b) + x_l` (행렬 가중치 W, O(d^2) 파라미터)
- DCN-v2가 더 표현력이 높고 실제 성능이 우수

### Stacked vs Parallel 구조
이 구현은 Parallel (병렬) 구조:
- Cross 경로와 Deep 경로가 독립적으로 실행
- 마지막에 concat으로 결합

Stacked 구조: Cross 출력을 Deep 입력으로 직렬 연결

## 관련 모델과의 비교

| 모델 | 교차항 학습 | 비선형성 | 피처 엔지니어링 | 사용 단계 |
|------|------------|----------|----------------|-----------|
| **DCN-v2** | 자동 (고차) | MLP | 불필요 | 랭킹 |
| **FM (Factorization Machine)** | 2차만 | 없음 | 불필요 | 랭킹 |
| **DeepFM** | 2차 (FM) | MLP | 불필요 | 랭킹 |
| **NCF** | 내적 | MLP | 불필요 | 랭킹 |
| **PLE-MTL** | 없음 | Expert MLP | 불필요 | 멀티태스크 |

### DCN-v2 vs FM
- FM: 2차 교차항만 학습 (O(kd) 시간)
- DCN-v2: Cross Layer 깊이에 따라 임의 고차 교차항 학습

### DCN-v2 vs DeepFM
- DeepFM: FM 컴포넌트 + DNN 병렬 (2차 교차항에 특화)
- DCN-v2: Cross Layer가 FM을 대체하며 더 높은 차수 교차 지원

### DCN-v2 vs FT-Transformer
- DCN-v2: 교차항을 명시적으로 모델링, 속도 빠름
- FT-Transformer: Self-Attention으로 암묵적 교차 학습, 피처 수가 많을수록 강력

### 참고 논문
- Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (WWW 2021)
- Wang et al., "Deep & Cross Network for Ad Click Predictions" (AdKDD 2017)
