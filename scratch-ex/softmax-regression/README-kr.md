# Softmax Regression (소프트맥스 회귀)

> 다중 분류를 위한 모델. Logistic Regression의 다중 클래스 확장.

## 실행 방법

```bash
cd scratch-ex/softmax-regression
python softmax_regression.py
```

## 핵심 개념

### 모델

```
z = X @ W + b           # (n, K) 각 클래스에 대한 점수
P(y=k|x) = softmax(z)   # 점수를 확률로 변환
```

### Softmax 함수

```
softmax(z_k) = exp(z_k) / Σ exp(z_j)
```

- 모든 출력의 합 = 1 (확률 분포)
- 가장 큰 값이 가장 높은 확률을 가짐

### 손실 함수: Cross-Entropy

```
L = -(1/n) * Σ Σ y_k * log(p_k)
```

- one-hot 인코딩된 정답과 예측 확률의 교차 엔트로피
- Binary Cross-Entropy의 다중 클래스 일반화

### 비교

| | Logistic | Softmax |
|---|---|---|
| 클래스 수 | 2 | K (2 이상) |
| 활성화 | sigmoid | softmax |
| 출력 | 스칼라 (0~1) | 벡터 (합=1) |
| 파라미터 | w: (d,1), b: (1,1) | W: (d,K), b: (1,K) |

K=2일 때 softmax는 logistic과 동치이다.

## 학습 데이터

합성 데이터: 2D 평면에서 3개 그룹의 점
- 클래스 0: 중심 (-2, 0)
- 클래스 1: 중심 (2, 0)
- 클래스 2: 중심 (0, 2)
- 각 100개, 총 300개 샘플
