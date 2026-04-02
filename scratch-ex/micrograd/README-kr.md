# Micrograd

> 스칼라 값 기반의 자동 미분(autograd) 엔진과 그 위에 구축된 신경망 라이브러리.
>
> 원본: [karpathy/micrograd](https://github.com/karpathy/micrograd)

## 실행 방법

```bash
cd scratch-ex/micrograd
python demo.py
```

의존성: `numpy`, `scikit-learn` (데이터셋 생성용)

## 파일 구조

| 파일 | 설명 |
|------|------|
| `engine.py` | `Value` 클래스 — 스칼라 autograd 엔진 |
| `nn.py` | `Neuron`, `Layer`, `MLP` — 신경망 구성 요소 |
| `demo.py` | make_moons 이진 분류 학습 예제 |

## 핵심 구성 요소

### 1. Value (engine.py)

스칼라 단위 자동 미분 엔진. microgpt의 `Value`와 같은 개념이지만 구현 방식이 다르다.

| 비교 | micrograd | microgpt |
|------|-----------|----------|
| 미분 저장 | `_backward` 클로저 | `_local_grads` 튜플 |
| 지원 연산 | `+`, `*`, `**`, `relu` | `+`, `*`, `**`, `relu`, `log`, `exp` |
| 디버깅 | `_op` 필드로 연산 추적 | 없음 |

**핵심 원리**: 각 연산이 실행될 때 (forward pass):
1. 결과값을 계산한다
2. 자식 노드를 기록한다
3. `_backward` 클로저에 로컬 미분 규칙을 저장한다

`backward()` 호출 시 (backward pass):
1. 위상 정렬(topological sort)로 계산 그래프를 순서화한다
2. 역순으로 순회하며 chain rule을 적용한다

### 2. Neural Network (nn.py)

계층적 구조:

```
MLP
 └── Layer (여러 개)
      └── Neuron (여러 개)
           └── Value (가중치, 편향)
```

- **Neuron**: `sum(w_i * x_i) + b` → ReLU (선택)
- **Layer**: 동일한 Neuron N개의 모음
- **MLP**: 여러 Layer를 순차적으로 연결

### 3. 학습 (demo.py)

- **데이터**: `make_moons` — 반달 모양의 2D 이진 분류 데이터셋 (100개 샘플)
- **모델**: MLP(2, [16, 16, 1]) — 입력 2차원, 은닉층 16-16, 출력 1차원
- **손실함수**: SVM max-margin loss + L2 정규화
- **옵티마이저**: SGD (learning rate decay 적용)
- **평가**: 정확도 (accuracy)

## microgpt와의 관계

micrograd는 **autograd 엔진**에 집중하고, microgpt는 그 위에 **Transformer 아키텍처**를 구축한다.

```
micrograd (autograd + MLP)
    ↓ Value 클래스 확장
microgpt (autograd + Transformer + Tokenizer + Adam)
```

학습 순서: **micrograd → microgpt** 를 추천한다. autograd를 먼저 이해하면 microgpt의 나머지 부분에 집중할 수 있다.
