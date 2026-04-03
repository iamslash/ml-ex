# Meta-Learners (메타 러너)

> S/T/X-Learner 세 가지 메타 학습 전략으로 CATE(개인별 처리 효과)를 추정하고 비교하는 인과추론 프레임워크.

## 실행 방법

```bash
cd causal-ex/meta-learners
pip install numpy scikit-learn
python meta_learners.py
```

**출력 예시:**
```
train: 1400, test: 600
treatment rate: 0.50
true ATE: 2.00

=== S-Learner ===
  ATE estimate: 1.87
  CATE RMSE: 0.7234

=== T-Learner ===
  ATE estimate: 1.95
  CATE RMSE: 0.6512

=== X-Learner ===
  ATE estimate: 1.98
  CATE RMSE: 0.5231
```

## 핵심 개념

### CATE와 ATE

- **CATE(x)** = E[Y(1) - Y(0) | X = x]: 공변량 X가 주어진 개인의 처리 효과
- **ATE** = E[τ(X)]: 전체 모집단 평균 처리 효과

이 예제의 이질적 처리 효과:
```python
tau(X) = 2 * (X[:,2] > 0) + X[:,3]  # 개인마다 다른 처리 효과
```

### S-Learner (Single Learner)

처리 변수 T를 일반 특성처럼 포함하여 단일 모델 학습:

```python
# 학습: X와 T를 합쳐서 Y 예측
model.fit([X | T], Y)

# 추론: T=1과 T=0 예측값의 차이
CATE(x) = model.predict([x, 1]) - model.predict([x, 0])
```

**단점**: 트리 기반 모델에서 T의 영향이 희석되어 처리 효과를 과소 추정하는 경향.

### T-Learner (Two Learner)

처리군과 대조군 각각 별도 모델:

```python
mu1.fit(X[T==1], Y[T==1])  # 처리군 모델
mu0.fit(X[T==0], Y[T==0])  # 대조군 모델

CATE(x) = mu1.predict(x) - mu0.predict(x)
```

**단점**: 처리 배정이 불균형(한쪽이 매우 적을 때)이면 한 모델의 분산이 커짐.

### X-Learner (Cross Learner)

T-Learner를 기반으로 잔차(imputed treatment effect)를 학습:

**Step 1**: T-Learner와 동일하게 `mu0`, `mu1` 학습

**Step 2**: 반사실적 잔차 계산:
```python
D1 = Y[T==1] - mu0.predict(X[T==1])  # 처리군: 실제 - 대조 예측
D0 = mu1.predict(X[T==0]) - Y[T==0]  # 대조군: 처리 예측 - 실제
```

**Step 3**: 잔차를 예측하는 모델 학습:
```python
tau1.fit(X[T==1], D1)
tau0.fit(X[T==0], D0)
```

**Step 4**: Propensity 가중 결합:
```python
e(x) = P(T=1|X=x)  # propensity score
CATE(x) = e(x) * tau1.predict(x) + (1-e(x)) * tau0.predict(x)
```

Propensity가 높은 영역은 `tau1` 추정을, 낮은 영역은 `tau0` 추정을 더 신뢰.

### 기저 모델: GradientBoostingRegressor

모든 메타 러너가 Gradient Boosting을 기저 모델로 사용. 메타 러너 프레임워크는 기저 모델에 무관하게 동작 (랜덤 포레스트, 신경망 등으로 교체 가능).

## 관련 모델과의 비교

| 방법 | 모델 수 | 처리 불균형 대응 | CATE 정밀도 | 복잡도 |
|------|--------|----------------|------------|--------|
| **S-Learner** | 1개 | 보통 | 낮음 (희석 편향) | 낮음 |
| **T-Learner** | 2개 | 약함 | 중간 | 낮음 |
| **X-Learner** | 5개 | 강함 | 높음 | 중간 |
| **R-Learner** | 2단계 잔차 | 중간 | 높음 | 중간 |
| **DR-Learner** | 2단계 DR | 강함 | 높음 | 높음 |
| **HydraNet** | 신경망 | 중간 | 높음 (비선형) | 높음 |

**언제 어떤 방법을 사용할까:**
- 처리 배정이 거의 균등하고 단순한 구조: T-Learner
- 처리 배정이 불균형하거나 희귀 처리군: X-Learner
- 대규모 데이터, 비선형 복잡 관계: HydraNet 또는 DR-Learner
- 빠른 기준선(baseline): S-Learner
