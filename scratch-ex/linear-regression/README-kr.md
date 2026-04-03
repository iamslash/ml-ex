# Linear Regression (선형 회귀)

> 가장 기본적인 지도학습 모델. 입력과 출력 사이의 선형 관계를 학습한다.

## 실행 방법

```bash
cd scratch-ex/linear-regression
python linear_regression.py
```

## 핵심 개념

### 모델

```
y = w * x + b
```

- `w` (가중치): 기울기
- `b` (편향): y절편

### 손실 함수: MSE (Mean Squared Error)

```
L = (1/n) * Σ(y_pred - y)²
```

예측값과 실제값의 차이를 제곱하여 평균. 값이 작을수록 좋다.

### 경사하강법 (Gradient Descent)

손실 함수의 그래디언트 방향의 반대로 파라미터를 갱신한다.

```
dL/dw = (2/n) * X^T @ (y_pred - y)
dL/db = (2/n) * Σ(y_pred - y)

w = w - lr * dL/dw
b = b - lr * dL/db
```

### 학습 루프

```
1. Forward:  y_pred = X @ w + b       (예측)
2. Loss:     L = mean((y_pred - y)²)  (손실 계산)
3. Backward: dw, db 계산              (그래디언트)
4. Update:   w -= lr * dw             (파라미터 갱신)
```

이 4단계 루프는 모든 신경망 학습의 기본 구조이다.

## 학습 데이터

코드에서 직접 생성하는 합성 데이터:
- `y = 3x + 7 + noise`
- 100개 샘플, noise는 정규분포 N(0, 0.5)

학습이 성공하면 w → 3, b → 7 에 수렴한다.
