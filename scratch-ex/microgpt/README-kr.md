# MicroGPT

> 순수 Python으로 구현한 GPT. 외부 의존성 없이 ~200줄로 학습과 추론을 모두 수행한다.
>
> 원본: [karpathy/microgpt gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## 실행 방법

```bash
cd scratch-ex/microgpt
python microgpt.py
```

`input.txt`가 없으면 자동으로 [names.txt](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)를 다운로드한다.

## 핵심 구성 요소

### 1. Value (Autograd 엔진)

스칼라 단위의 자동 미분 엔진. 각 연산마다 계산 그래프를 구성하고 `backward()`로 역전파를 수행한다.

- `data`: forward pass에서 계산된 스칼라 값
- `grad`: loss에 대한 이 노드의 미분값 (backward pass에서 계산)
- `_children`: 계산 그래프에서의 자식 노드들
- `_local_grads`: 자식 노드에 대한 로컬 미분값 (chain rule 적용에 사용)

### 2. Tokenizer

문자 단위(character-level) 토크나이저. 데이터셋의 고유 문자 → 정수 매핑.

- `uchars`: 고유 문자 목록 (정렬됨)
- `BOS`: 시퀀스 시작/끝을 나타내는 특수 토큰

### 3. GPT 모델 구조

GPT-2 기반이며 다음과 같은 차이가 있다:

| GPT-2 | MicroGPT |
|-------|----------|
| LayerNorm | RMSNorm |
| bias 사용 | bias 없음 |
| GeLU | ReLU |

구성:
- **Token Embedding (`wte`)**: 토큰 → 벡터
- **Position Embedding (`wpe`)**: 위치 → 벡터
- **Multi-Head Attention**: Q, K, V 변환 → 스케일드 닷 프로덕트 어텐션 → 출력 프로젝션
- **MLP**: fc1 (확장) → ReLU → fc2 (축소)
- **Residual Connection**: 각 블록의 입력을 출력에 더함

### 4. Adam Optimizer

파라미터 업데이트 규칙:

- 1차 모멘트 (`m`): 그래디언트의 이동 평균
- 2차 모멘트 (`v`): 그래디언트 제곱의 이동 평균
- bias correction 적용 후 파라미터 업데이트
- linear learning rate decay 사용

### 5. 추론 (Inference)

- temperature 파라미터로 생성 다양성 조절 (낮을수록 보수적)
- `random.choices`로 확률 기반 샘플링
- BOS 토큰이 생성되면 시퀀스 종료

## 하이퍼파라미터

| 이름 | 값 | 설명 |
|------|-----|------|
| `n_layer` | 1 | 트랜스포머 레이어 수 |
| `n_embd` | 16 | 임베딩 차원 |
| `block_size` | 16 | 최대 컨텍스트 길이 |
| `n_head` | 4 | 어텐션 헤드 수 |
| `num_steps` | 1000 | 학습 스텝 수 |
| `learning_rate` | 0.01 | 초기 학습률 |
| `temperature` | 0.5 | 추론 시 생성 온도 |
