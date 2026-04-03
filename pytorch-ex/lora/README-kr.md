# LoRA (Low-Rank Adaptation)

> 사전학습 모델의 가중치를 동결하고, 작은 low-rank 행렬만 학습하는 파라미터 효율적 fine-tuning.

## 실행 방법

```bash
cd pytorch-ex/lora
python lora.py
```

## 핵심 개념

### 아이디어

```
W_new = W_frozen + A @ B
```

- `W_frozen`: 사전학습된 가중치 (동결, 학습 안 함)
- `A`: (in, rank) 행렬 (학습)
- `B`: (rank, out) 행렬 (학습)
- `rank << min(in, out)`

### 왜 작동하는가

fine-tuning 시 가중치 변화량 ΔW는 보통 low-rank 구조를 가진다.
즉, 전체 가중치를 바꿀 필요 없이 작은 행렬 두 개로 변화량을 표현할 수 있다.

### 파라미터 효율성

| | Full Fine-tuning | LoRA (rank=4) |
|---|---|---|
| 학습 파라미터 | 전체 | ~5-10% |
| 메모리 | 전체 그래디언트 | 작은 행렬만 |
| 모델 저장 | 전체 복사본 | 작은 adapter만 |
| 태스크 전환 | 별도 모델 | adapter 교체 |

### 이 코드에서의 흐름

```
1. MLP를 sin(x) 태스크로 사전학습
2. Linear → LoRALinear로 교체 (원래 가중치 동결)
3. cos(x) 태스크로 LoRA fine-tuning
4. 전체 파라미터 중 일부만 학습하면서도 새 태스크 수행
```

### 실제 활용

- LLM fine-tuning (GPT, LLaMA 등)
- Stable Diffusion 스타일 적응
- 하나의 기본 모델에 여러 LoRA adapter를 교체하며 사용
