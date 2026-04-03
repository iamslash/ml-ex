# HydraNet Uplift Model (하이드라넷 업리프트 모델)

> 공유 표현과 처리별 헤드로 다중 처리 효과(CATE)를 동시에 추정하는 인과추론 딥러닝 모델.

## 실행 방법

```bash
cd causal-ex/hydranet
pip install torch numpy
python hydranet.py
```

**출력 예시:**
```
samples: 3000, features: 8, treatments: 3
epoch  20 | loss 0.8234 | CATE_A=0.87 (true~1.0) | CATE_B=1.41 (true~1.5)
epoch  40 | loss 0.6123 | CATE_A=0.95 (true~1.0) | CATE_B=1.48 (true~1.5)
...
  X2>0 (Treatment A target):
    CATE_A: +=1.82, -=0.12
    CATE_B: +=2.91, -=0.08
```

## 핵심 개념

### CATE (Conditional Average Treatment Effect)

개인별 처리 효과의 조건부 기대값:

```
CATE(x) = E[Y(t) - Y(0) | X = x]
```

- `Y(t)`: 처리 t를 받았을 때의 잠재 결과
- `Y(0)`: 대조군(무처리) 결과
- 같은 처리라도 공변량 X에 따라 효과가 다름 (이질적 처리 효과)

이 예제의 진짜 CATE:
- Treatment A: `X[:,2] > 0`이면 +2.0, 아니면 0
- Treatment B: `X[:,3] > 0`이면 +3.0, 아니면 0

### HydraNet 구조

```
입력 X (8차원)
   -> Shared Trunk: Linear(8,64)->ReLU->Linear(64,64)->ReLU
   -> Head_0: Linear(64,32)->ReLU->Linear(32,1)  [대조군 결과]
   -> Head_1: Linear(64,32)->ReLU->Linear(32,1)  [처리 A 결과]
   -> Head_2: Linear(64,32)->ReLU->Linear(32,1)  [처리 B 결과]
   -> Propensity Head: Linear(64, 3)              [처리 배정 확률]
```

- **공유 트렁크(Shared Trunk)**: 모든 처리 조건에서 공통된 표현 학습
- **처리별 헤드(Treatment Head)**: 각 처리에 특화된 결과 예측
- **Propensity Head**: 처리 배정 확률 예측 (보조 태스크)

### DR Loss (Doubly Robust Loss)

아웃컴 회귀와 역확률 가중치(IPW)를 결합:

```
DR Estimator = E[μ_t(X) + (Y - μ_t(X)) * I(T=t) / e(t|X)]
```

구현에서는 두 가지 손실을 결합:

```python
# 1. Outcome regression (관측된 처리에 대해서만)
loss += MSELoss(outcomes[t][T==t], Y[T==t])

# 2. Propensity loss (보조 태스크, 가중치 0.1)
loss += 0.1 * CrossEntropyLoss(prop_logits, T)
```

DR 추정량은 아웃컴 모델 또는 propensity 모델 중 하나만 올바르면 일관 추정량이 됨 (이중 견고성).

### 잠재 결과 프레임워크 (Potential Outcomes)

관측 데이터의 근본적 문제: 한 개인은 하나의 처리만 받을 수 있음. HydraNet은 각 헤드가 관측되지 않은 반사실적 결과를 예측하도록 학습.

## 관련 모델과의 비교

| 모델 | 처리 수 | 구조 | 특징 |
|------|--------|------|------|
| **HydraNet (이 예제)** | 다중 | 공유 표현 + 다중 헤드 | GPU 가속, 비선형 CATE |
| **S-Learner** | 단일/다중 | 단일 모델 | 단순, 처리 효과 축소 편향 |
| **T-Learner** | 단일 | 처리별 독립 모델 | 간단, 소수 처리군에서 분산 큼 |
| **X-Learner** | 단일 | 2단계 메타 학습 | 불균형 처리 배정에 강건 |
| **DragonNet** | 단일 | 공유 표현 + DR | HydraNet의 단일 처리 버전 |

**HydraNet의 핵심 장점:**
- 처리 수가 늘어도 공유 표현 덕분에 파라미터 효율적
- Propensity 보조 태스크가 표현 학습을 개선 (정규화 효과)
- 배치 처리로 모든 처리 조건을 동시에 추론 가능
