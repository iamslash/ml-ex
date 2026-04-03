# LSTM-FCN (LSTM + Fully Convolutional Network)

> LSTM 경로와 1D CNN 경로를 병렬로 실행한 뒤 결합하여 시계열 분류 성능을 높이는 하이브리드 아키텍처

## 실행 방법

```bash
cd timeseries-ex/lstm-fcn
python lstm_fcn.py
```

의존성: `torch`, `numpy`

3개 클래스(sin/sawtooth/step 패턴) 각 200샘플, 시퀀스 길이 50의 합성 데이터로 분류를 학습한다.

## 핵심 개념

### 병렬 이중 경로 아키텍처

```
입력 (batch, seq_len)
      |               |
      v               v
 LSTM 경로        FCN(CNN) 경로
 (seq_len, 1)     (1, seq_len)
      |               |
 마지막 은닉          Global Average Pooling
 (batch, 64)          (batch, 128)
      |               |
      +-------+-------+
              |
         Concat (batch, 192)
              |
           Linear -> 클래스 (3)
```

두 경로가 서로 다른 측면을 학습한다.
- **LSTM**: 시간적 장기 의존성, 순서 정보
- **FCN**: 로컬 패턴, 위치 불변 특징

### FCN 경로 상세
```python
Conv1d(1, 128, kernel_size=8)    # 큰 커널: 긴 패턴
Conv1d(128, 256, kernel_size=5)  # 중간 커널
Conv1d(256, 128, kernel_size=3)  # 작은 커널: 세밀한 패턴
```

각 Conv 레이어에 BatchNorm + ReLU 적용. 피라미드 구조로 다양한 스케일의 특징 포착.

### Global Average Pooling (GAP)
```python
cnn_feat = c.mean(dim=2)  # (batch, 128, seq_len) -> (batch, 128)
```

시퀀스 길이 전체를 평균하여 고정 크기 벡터로 변환한다.
- Global Max Pooling 대비 노이즈에 강건
- FC 레이어 파라미터 수 대폭 감소
- 가변 길이 시퀀스에 자연스럽게 적용 가능

### LSTM 입력 차원 변환
```python
x_lstm = x.unsqueeze(-1)  # (batch, seq_len) -> (batch, seq_len, 1)
x_cnn  = x.unsqueeze(1)   # (batch, seq_len) -> (batch, 1, seq_len)
```

같은 입력을 LSTM은 `(batch, time, feature)`, CNN은 `(batch, channel, time)` 형태로 달리 해석한다.

### 드롭아웃
```python
self.lstm_dropout = nn.Dropout(0.2)
```

LSTM 출력에만 드롭아웃 적용. FCN은 BatchNorm이 정규화 역할 수행.

## 관련 모델과의 비교

| 항목 | LSTM-FCN | 순수 LSTM | 순수 CNN | Transformer |
|------|----------|-----------|----------|-------------|
| 장기 의존성 | LSTM 경로로 처리 | 강점 | 약함 | 어텐션으로 처리 |
| 로컬 패턴 | FCN 경로로 처리 | 약함 | 강점 | 제한적 |
| 위치 불변성 | GAP으로 달성 | 없음 | GAP으로 달성 | 없음 |
| 파라미터 수 | 중간~많음 | 적음 | 적음 | 매우 많음 |
| 학습 속도 | 중간 | 빠름 | 빠름 | 느림 |

### LSTM-FCN vs ResNet (시계열)
- **LSTM-FCN**: LSTM 경로로 순서 정보 명시적 포착
- **ResNet**: 잔차 연결로 깊은 CNN 학습, 순서 정보는 묵시적

### LSTM-FCN vs InceptionTime
- **LSTM-FCN**: 두 가지 경로 병렬 결합
- **InceptionTime**: 다양한 커널 크기의 Conv를 병렬 결합(Inception 모듈), LSTM 없음

### GAP vs Flatten
| 방법 | 파라미터 | 가변 길이 | 과적합 위험 |
|------|---------|---------|-----------|
| GAP (이 코드) | 적음 | 가능 | 낮음 |
| Flatten | 많음 | 불가 | 높음 |
