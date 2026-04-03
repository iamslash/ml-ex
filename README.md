# ML Hands-on

ML 모델을 쉬운 것부터 어려운 것까지 직접 구현하고 실습하는 리포지토리.

## 학습 로드맵

### Phase 1: 기초 (From Scratch)

| # | 모델 | 디렉토리 | 방식 |
|---|------|----------|------|
| 1 | Linear Regression | `scratch-ex/linear-regression` | numpy |
| 2 | Logistic Regression | `scratch-ex/logistic-regression` | numpy |
| 3 | Softmax Regression | `scratch-ex/softmax-regression` | numpy |
| 4 | kNN | `scratch-ex/knn` | numpy |
| 5 | Decision Tree | `scratch-ex/decision-tree` | numpy |

### Phase 2: 신경망 기초 (From Scratch)

| # | 모델 | 디렉토리 | 방식 |
|---|------|----------|------|
| 6 | MLP (Micrograd) | `scratch-ex/micrograd` | pure python |
| 7 | CNN (최소 구현) | `scratch-ex/cnn-minimal` | numpy |
| 8 | Self-Attention | `scratch-ex/self-attention` | numpy |
| 9 | GPT (MicroGPT) | `scratch-ex/microgpt` | pure python |

### Phase 3: PyTorch 실습

| # | 모델 | 디렉토리 | 핵심 |
|---|------|----------|------|
| 10 | CNN | `pytorch-ex/cnn` | MNIST 분류, DataLoader |
| 11 | RNN/LSTM | `pytorch-ex/rnn-lstm` | 시퀀스 모델링 |
| 12 | Transformer | `pytorch-ex/transformer` | 인코더-디코더, 어텐션 |
| 13 | BERT Fine-tuning | `pytorch-ex/bert-finetune` | 사전학습 모델 활용 |
| 14 | ViT | `pytorch-ex/vit` | 이미지 패치 + 트랜스포머 |
| 15 | wav2vec2 | `pytorch-ex/wav2vec2` | 음성 표현 학습 |

### Phase 4: 생성 모델 (PyTorch)

| # | 모델 | 디렉토리 | 핵심 |
|---|------|----------|------|
| 16 | Autoencoder | `pytorch-ex/autoencoder` | 압축/복원, 잠재 공간 |
| 17 | VAE | `pytorch-ex/vae` | 확률적 생성 |
| 18 | Diffusion | `pytorch-ex/diffusion` | 노이즈 제거 생성 |

### Phase 5: 강화학습

| # | 모델 | 디렉토리 | 핵심 |
|---|------|----------|------|
| 19 | Q-Learning | `scratch-ex/q-learning` | 테이블 기반 RL |
| 20 | DQN | `pytorch-ex/dqn` | 신경망 기반 RL |

### Phase 6: 응용

| # | 모델 | 디렉토리 | 핵심 |
|---|------|----------|------|
| 21 | LoRA | `pytorch-ex/lora` | 파라미터 효율적 fine-tuning |

## 설치

```bash
# 기본 (scratch-ex 전체, pytorch-ex 일부)
pip install numpy torch torchvision scikit-learn matplotlib

# BERT, ViT, wav2vec2
pip install transformers

# wav2vec2 추가
pip install torchaudio
```

## 참고 자료

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [karpathy/microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection)
