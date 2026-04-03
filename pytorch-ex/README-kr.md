# PyTorch 실습

> PyTorch를 활용한 모델 구현, 학습, 평가. scratch로 원리를 이해한 뒤 실용적 구현으로 확장한다.

## 학습 순서

| # | 모델 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | CNN | [cnn](cnn/) | MNIST/CIFAR 분류, DataLoader, 학습 루프 |
| 2 | RNN/LSTM | [rnn-lstm](rnn-lstm/) | 시퀀스 모델링, 텍스트 분류 |
| 3 | Transformer | [transformer](transformer/) | 인코더-디코더, 어텐션 |
| 4 | BERT Fine-tuning | [bert-finetune](bert-finetune/) | 사전학습 모델 활용, 감성 분류 |
| 5 | ViT | [vit](vit/) | 이미지 패치 → 트랜스포머 |
| 6 | wav2vec2 | [wav2vec2](wav2vec2/) | 음성 표현 학습 |
| 7 | Autoencoder | [autoencoder](autoencoder/) | 인코더-디코더, 잠재 공간 |
| 8 | VAE | [vae](vae/) | 확률적 잠재 공간, reparameterization |
| 9 | Diffusion | [diffusion](diffusion/) | 노이즈 추가/제거, denoising |
| 10 | DQN | [dqn](dqn/) | 신경망 + Q-Learning |
| 11 | LoRA | [lora](lora/) | 파라미터 효율적 fine-tuning |
