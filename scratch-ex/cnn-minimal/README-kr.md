# CNN Minimal (합성곱 신경망 최소 구현)

> 순수 numpy로 Conv2D, MaxPool, FC를 구현한 최소한의 CNN.

## 실행 방법

```bash
cd scratch-ex/cnn-minimal
python cnn_minimal.py
```

주의: 수치 미분을 사용하므로 실행 시간이 수 분 걸릴 수 있다.

## 핵심 개념

### Conv2D (합성곱)

```
output[i,j] = Σ image[i:i+kH, j:j+kW] * kernel
```

- 작은 필터(커널)를 이미지 위에서 슬라이딩하며 내적 계산
- 각 필터는 하나의 특징(가로선, 세로선 등)을 감지
- valid padding: 출력 크기 = (H-kH+1, W-kW+1)

### Max Pooling

```
output[i,j] = max(image[i*s:(i+1)*s, j*s:(j+1)*s])
```

- 영역 내 최대값만 선택 → 공간 축소, 위치 불변성

### CNN 파이프라인

```
입력 이미지 (5x5)
  → Conv2D (3x3 커널, 4개 필터) → (3x3) x 4
  → ReLU
  → MaxPool (3x3) → (1x1) x 4
  → Flatten → (4,)
  → FC → (3,)
  → Softmax → 확률
```

### 왜 수치 미분인가?

scratch CNN에서 backprop을 구현하려면 conv/pool의 역전파가 필요하다.
이 코드는 원리 이해에 집중하기 위해 수치 미분 `(f(x+h) - f(x-h)) / 2h`을 사용한다.
PyTorch 버전에서는 autograd가 이를 자동으로 처리한다.

### MLP와의 비교

| | MLP | CNN |
|---|---|---|
| 입력 | 1D 벡터 | 2D 이미지 (구조 유지) |
| 가중치 | 전체 연결 | 필터 공유 (파라미터 효율) |
| 특징 | 위치 정보 무시 | 공간적 패턴 감지 |

## 학습 데이터

합성 5x5 이미지: 3가지 패턴
- 클래스 0: 수평선
- 클래스 1: 수직선
- 클래스 2: 대각선
- train 240개, test 60개
