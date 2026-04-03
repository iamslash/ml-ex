# Image Captioning (이미지 캡셔닝)

> 이미지 특성을 LSTM 디코더에 입력하여 텍스트 시퀀스를 생성하는 모델.

## 실행 방법

```bash
cd multimodal-ex/image-captioning
python image_captioning.py
```

## 핵심 개념

### 구조

```
이미지 → Image Encoder → 초기 은닉 상태 (h0)
                                ↓
[BOS] → Embedding → LSTM(h0) → 토큰1 예측
토큰1 → Embedding → LSTM     → 토큰2 예측
...                           → [EOS] 종료
```

### Teacher Forcing

학습 시 이전 스텝의 정답 토큰을 다음 입력으로 사용. 추론 시에는 자기 자신의 예측을 입력으로 사용 (autoregressive).

### CLIP과의 비교

| | CLIP | Image Captioning |
|---|---|---|
| 출력 | 임베딩 (매칭) | 텍스트 시퀀스 (생성) |
| 과제 | 이미지-텍스트 유사도 | 이미지 → 텍스트 |
| 모델 | 인코더만 | 인코더 + 디코더 |
