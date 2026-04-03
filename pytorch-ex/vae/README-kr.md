# VAE (Variational Autoencoder)

> 잠재 공간을 확률 분포로 모델링하여 새로운 데이터를 생성할 수 있는 생성 모델.

## 실행 방법

```bash
cd pytorch-ex/vae
python vae.py
```

## 핵심 개념

### Autoencoder와의 차이

| | Autoencoder | VAE |
|---|---|---|
| 잠재 공간 | 고정된 벡터 | 확률 분포 (mu, sigma) |
| 생성 | 어려움 | 가능 (분포에서 샘플링) |
| 손실 | MSE만 | Reconstruction + KL divergence |

### Reparameterization Trick

```
z = mu + std * eps    (eps ~ N(0, 1))
```

- 샘플링은 미분 불가 → 역전파 불가능
- eps를 외부에서 샘플링하고, z를 mu와 std의 함수로 표현
- 이제 mu, std에 대해 역전파 가능

### 손실 함수

```
Loss = Reconstruction Loss + KL Divergence
```

- **Reconstruction Loss**: 입력과 복원의 차이 (BCE)
- **KL Divergence**: 잠재 분포 q(z|x)와 표준 정규분포 N(0,1)의 차이
  - 잠재 공간을 매끄럽고 규칙적으로 만든다
  - 이것이 새로운 이미지 생성을 가능하게 하는 핵심

### 생성 과정

```
z ~ N(0, 1)  →  Decoder  →  새로운 이미지
```

학습 후 표준 정규분포에서 z를 샘플링하면 그럴듯한 이미지가 나온다.
