# PLE-MTL (Progressive Layered Extraction, 점진적 계층 추출 멀티태스크 학습)

> 공유/전용 Expert 네트워크와 Gating으로 다중 태스크를 동시에 학습하며, SwiGLU 활성화와 불확실성 기반 손실 가중치를 적용하는 모델

## 실행 방법

```bash
cd recsys-ex/ple-mtl
python ple_mtl.py
```

의존성: `torch`

출력 예시:
- 에폭별 loss, Task A/B 정확도, 불확실성 가중치 (10 에폭마다)

## 핵심 개념

### PLE 아키텍처

```
Input x (16차원)
       |
  +----+------------------------+
  |                             |
Shared Experts (2개)    Task A Experts (2개)   Task B Experts (2개)
  E_s1, E_s2                E_a1, E_a2              E_b1, E_b2
  |                             |                         |
  +----------+-----------------+         +---------------+
             |                           |
        Gate_A (softmax)            Gate_B (softmax)
        [shared + task_a]           [shared + task_b]
             |                           |
        weighted sum               weighted sum
             |                           |
        Task A output              Task B output
             |                           |
        Tower_A -> logit_A          Tower_B -> logit_B
```

각 PLE Layer를 `n_layers=2`번 쌓아 계층적 추출 수행

### SwiGLU 활성화

```python
def forward(self, x):
    return self.W(x) * torch.sigmoid(self.V(x))
```

- LLM(LLaMA, PaLM 등)에서 사용되는 활성화 함수
- Swish(x) * Linear(x) 형태의 게이팅
- ReLU 대비 학습 안정성과 표현력 향상

### Gating Network

```python
weights = softmax(gate_linear(x))   # (batch, n_experts)
out = sum(weights * expert_outputs)  # 가중 합산
```

- 입력에 따라 각 Expert의 기여도를 동적으로 결정
- Task A의 Gate는 공유 Expert + Task A 전용 Expert만 참조
- Task 간 불필요한 간섭 차단

### Uncertainty Loss (Kendall et al.)

```python
class UncertaintyLoss(nn.Module):
    def __init__(self, n_tasks):
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        precision = exp(-log_var)
        total = precision * loss + log_var
```

- 태스크별 가중치를 학습 가능한 파라미터로 자동 조정
- `log_var`가 커지면 해당 태스크의 가중치가 낮아짐
- 수동으로 태스크 가중치를 조정할 필요 없음

### 합성 태스크 설계

```python
y_a = (X[:, :8].sum(dim=1) > 0)   # 피처 0-7 의존
y_b = (X[:, 4:].sum(dim=1) > 0)   # 피처 4-15 의존
```

피처 4-7은 두 태스크가 공유 → 공유 Expert가 이를 담당하는지 검증

## 관련 모델과의 비교

| 모델 | 공유 구조 | 태스크 분리 | 부정적 전이 방지 | 복잡도 |
|------|-----------|------------|-----------------|--------|
| **PLE** | 공유 + 전용 Expert | 완전 | 강함 | 높음 |
| **MMOE** | 공유 Expert만 | Gate | 중간 | 중간 |
| **Shared Bottom** | 완전 공유 | 태스크 타워만 | 약함 | 낮음 |
| **Cross Stitch** | 파라미터 수준 | 선형 결합 | 중간 | 중간 |

### PLE vs MMOE (Multi-gate Mixture of Experts)
- MMOE: 모든 태스크가 동일한 Expert 풀을 공유
- PLE: 공유 Expert + 태스크 전용 Expert를 분리 구성
- PLE가 부정적 전이(negative transfer) 억제에 더 효과적

### PLE vs Shared Bottom
- Shared Bottom: 하나의 공유 네트워크 + 각 태스크 타워
- 태스크 간 갈등이 크면 공유 표현이 왜곡됨 (seesaw problem)
- PLE는 전용 Expert로 이 문제를 해결

### 실제 사용 사례
- Tencent 광고 시스템: CTR + CVR 동시 예측
- 커머스: 클릭률 + 구매율 + 리뷰 작성률 동시 최적화

### 참고 논문
- Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations" (RecSys 2020)
- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
