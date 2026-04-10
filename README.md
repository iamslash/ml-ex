# ML Hands-on

ML 모델을 쉬운 것부터 어려운 것까지 직접 구현하고 실습하는 리포지토리.

## 구조

```
ml-ex/
├── scratch-ex/       원리 체득 (순수 Python/numpy)
├── pytorch-ex/       핵심 모델 (PyTorch)
├── recsys-ex/        추천 시스템
├── search-ex/        검색/랭킹
├── causal-ex/        인과추론
├── safety-ex/        AI 안전
├── timeseries-ex/    시계열/이상탐지
├── graph-ex/         그래프 신경망
├── speech-ex/        음성
├── multimodal-ex/    멀티모달
├── agent-ex/         LLM 에이전트 패턴
└── triton-ex/        Triton
```

## 학습 로드맵

### Phase 1: 기초 (From Scratch)

| # | 모델 | 디렉토리 |
|---|------|----------|
| 1 | Linear Regression | `scratch-ex/linear-regression` |
| 2 | Logistic Regression | `scratch-ex/logistic-regression` |
| 3 | Softmax Regression | `scratch-ex/softmax-regression` |
| 4 | kNN | `scratch-ex/knn` |
| 5 | Decision Tree | `scratch-ex/decision-tree` |

### Phase 2: 신경망 기초 (From Scratch)

| # | 모델 | 디렉토리 |
|---|------|----------|
| 6 | MLP (Micrograd) | `scratch-ex/micrograd` |
| 7 | CNN (최소 구현) | `scratch-ex/cnn-minimal` |
| 8 | Self-Attention | `scratch-ex/self-attention` |
| 9 | GPT (MicroGPT) | `scratch-ex/microgpt` |

### Phase 3: PyTorch 핵심

| # | 모델 | 디렉토리 |
|---|------|----------|
| 10 | CNN (MNIST) | `pytorch-ex/cnn` |
| 11 | RNN/LSTM | `pytorch-ex/rnn-lstm` |
| 12 | Transformer | `pytorch-ex/transformer` |
| 13 | BERT Fine-tuning | `pytorch-ex/bert-finetune` |
| 14 | ViT | `pytorch-ex/vit` |

### Phase 4: 생성 모델

| # | 모델 | 디렉토리 |
|---|------|----------|
| 15 | Autoencoder | `pytorch-ex/autoencoder` |
| 16 | VAE | `pytorch-ex/vae` |
| 17 | Diffusion | `pytorch-ex/diffusion` |

### Phase 5: 강화학습

| # | 모델 | 디렉토리 |
|---|------|----------|
| 18 | Q-Learning | `scratch-ex/q-learning` |
| 19 | DQN | `pytorch-ex/dqn` |

### Phase 6: 응용

| # | 모델 | 디렉토리 |
|---|------|----------|
| 20 | LoRA | `pytorch-ex/lora` |

### Phase 7: 도메인별 실습

| 도메인 | 디렉토리 | 내용 |
|--------|----------|------|
| 추천 | `recsys-ex/` | MF, Two-Tower, NCF, DCN, PLE-MTL, Item2Vec, FT-Transformer, Offline RL |
| 검색 | `search-ex/` | BM25, Semantic Search, Re-ranker |
| 인과추론 | `causal-ex/` | HydraNet, Meta-Learners (S/T/X-Learner) |
| 안전 | `safety-ex/` | Toxicity Classifier, Reward Model, DPO |
| 시계열 | `timeseries-ex/` | Anomaly Autoencoder, Forecasting, LSTM-FCN |
| 그래프 | `graph-ex/` | GCN, GAT |
| 음성 | `speech-ex/` | Audio Classifier, wav2vec2 (참고 예제) |
| 멀티모달 | `multimodal-ex/` | CLIP, Image Captioning |
| 에이전트 | `agent-ex/` | Memory Schema, Reflexion, Memory Retrieval, Governance, Generation-Reflection, Iterative Refinement, LLM-TDD, Skill Library, Self-Correcting Agent |

## 설치

```bash
# 기본 (scratch-ex, pytorch-ex 대부분)
pip install numpy torch torchvision scikit-learn matplotlib

# BERT, ViT
pip install transformers

# wav2vec2
pip install torchaudio
```

## 참고 자료

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [karpathy/microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection)
