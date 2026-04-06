# Audio Classifier (음성 분류)

> Mel spectrogram을 입력으로 받는 CNN 기반 음성 명령어 분류기.

## 실행 방법

```bash
cd speech-ex/audio-classifier
python audio_classifier.py
```

GPU, 외부 데이터 다운로드 불필요. `torch`, `numpy`만 필요.

## 핵심 개념

### Mel Spectrogram

음성 파형을 시간-주파수 2D 표현으로 변환:

```
x축: 시간 프레임 (30 프레임)
y축: Mel 주파수 bins (20 bins)
값: 에너지 크기
```

이미지처럼 2D CNN으로 처리 가능.

### 모델 구조

```
Mel Spectrogram (1, 20, 30)
  → Conv2d(1→16) + BN + ReLU + MaxPool  → (16, 10, 15)
  → Conv2d(16→32) + BN + ReLU + MaxPool → (32, 5, 7)
  → Conv2d(32→64) + BN + ReLU + GAP     → (64, 1, 1)
  → Flatten → Linear(64→32) → Linear(32→3)
```

### GAP (Global Average Pooling)

`AdaptiveAvgPool2d((1, 1))`로 공간 차원을 제거. FC 대비 파라미터 수 대폭 감소.

### wav2vec2와의 비교

| | Audio CNN (이 코드) | wav2vec2 |
|---|---|---|
| 입력 | Mel spectrogram (2D) | 원시 파형 (1D) |
| 사전학습 | 없음 | 자기지도 학습 |
| 데이터 | 합성 (외부 불필요) | 실제 음성 (수 GB) |
| GPU | 불필요 | 권장 |
| 학습 목적 | CNN으로 오디오 분류 원리 | 사전학습 모델 fine-tuning |

## 학습 데이터

합성 Mel spectrogram: 3가지 명령어 패턴
- "yes": 저주파 + 고주파 이중 피크
- "no": 중주파 단일 피크, 짧은 지속
- "stop": 넓은 대역, 긴 지속
