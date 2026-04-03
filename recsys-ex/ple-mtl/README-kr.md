# PLE Multi-Task Learning (다중 과제 학습)

> 공유/전용 expert 네트워크와 게이팅으로 다중 과제를 동시에 학습하는 모델.

## 실행 방법

```bash
cd recsys-ex/ple-mtl
python ple_mtl.py
```

## 핵심 개념

### PLE (Progressive Layered Extraction)

```
입력 → 공유 Expert ──┐
                      ├→ Gate(task A) → Task A Tower → 예측 A
입력 → Task A Expert ─┘
                      ├→ Gate(task B) → Task B Tower → 예측 B
입력 → Task B Expert ─┘
```

- 공유 expert: 태스크 간 공통 패턴 학습
- 전용 expert: 태스크별 고유 패턴 학습
- Gate: 입력에 따라 expert 가중치를 동적으로 결정

### SwiGLU 활성화

```
SwiGLU(x) = W(x) * sigmoid(V(x))
```

LLM(LLaMA 등)에서 사용되는 현대적 활성화 함수. ReLU보다 표현력이 높다.

### Uncertainty Loss

```
L = Σ (1/2σ²_t) * L_t + log(σ_t)
```

태스크별 가중치를 자동으로 학습. 어려운 태스크에 더 큰 가중치를 부여.
