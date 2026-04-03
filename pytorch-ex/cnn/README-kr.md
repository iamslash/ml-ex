# CNN (PyTorch)

> PyTorch로 구현한 합성곱 신경망. MNIST 손글씨 숫자 분류.

## 실행 방법

```bash
cd pytorch-ex/cnn
python cnn_mnist.py
```

의존성: `torch`, `torchvision`

## 모델 구조

```
입력 (1, 28, 28)
  → Conv2d(1→16, 3x3) → ReLU → MaxPool(2x2)   → (16, 14, 14)
  → Conv2d(16→32, 3x3) → ReLU → MaxPool(2x2)  → (32, 7, 7)
  → Flatten                                      → (1568,)
  → Linear(1568→128) → ReLU                     → (128,)
  → Linear(128→10)                               → (10,)
```

## scratch 버전과의 비교

| | scratch (cnn-minimal) | PyTorch (이 코드) |
|---|---|---|
| 데이터 | 합성 5x5 패턴 | MNIST 28x28 |
| 미분 | 수치 미분 | autograd (자동) |
| 최적화 | SGD (수동) | Adam |
| DataLoader | 없음 | 배치 자동 처리 |
| GPU | 불가 | 지원 |

## PyTorch 학습 루프 패턴

```python
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()      # 그래디언트 초기화
        output = model(data)       # forward
        loss = criterion(output, target)  # 손실 계산
        loss.backward()            # backward
        optimizer.step()           # 파라미터 갱신

    model.eval()
    with torch.no_grad():          # 평가 시 그래디언트 불필요
        # 테스트 정확도 계산
```

이 패턴은 거의 모든 PyTorch 모델에서 동일하게 사용된다.
