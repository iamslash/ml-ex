# Logistic Regression (로지스틱 회귀)

> 이진 분류를 위한 기본 모델. 선형 회귀의 출력에 sigmoid를 적용하여 확률로 변환한다.

## 실행 방법

```bash
cd scratch-ex/logistic-regression
python logistic_regression.py
```

## 핵심 개념

### 모델

```
z = x @ w + b
P(y=1|x) = sigmoid(z) = 1 / (1 + e^(-z))
```

- sigmoid는 실수 → (0, 1) 범위로 변환
- 출력이 0.5 이상이면 클래스 1, 미만이면 클래스 0

### 손실 함수: Binary Cross-Entropy

```
L = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
```

- y=1일 때: -log(p) → p가 1에 가까울수록 loss 작음
- y=0일 때: -log(1-p) → p가 0에 가까울수록 loss 작음

### Linear Regression과의 비교

| | Linear Regression | Logistic Regression |
|---|---|---|
| 출력 | 연속값 (실수) | 확률 (0~1) |
| 활성화 | 없음 | sigmoid |
| 손실 함수 | MSE | Binary Cross-Entropy |
| 용도 | 회귀 | 이진 분류 |

## 학습 데이터

합성 데이터: 2D 평면에서 두 그룹의 점
- 그룹 0: 중심 (-1, -1)
- 그룹 1: 중심 (1, 1)
- 각 100개, 총 200개 샘플
