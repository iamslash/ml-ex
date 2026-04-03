# RNN/LSTM (PyTorch)

> LSTM을 사용한 시퀀스 분류. 텍스트 감성 분류 유사 과제.

## 실행 방법

```bash
cd pytorch-ex/rnn-lstm
python rnn_lstm.py
```

## 핵심 개념

### LSTM (Long Short-Term Memory)

RNN의 장기 의존성 문제를 해결하기 위한 구조.

```
입력 (batch, seq_len)
  → Embedding      → (batch, seq_len, embed_dim)
  → LSTM           → output: (batch, seq_len, hidden_dim)
                      h_n: (1, batch, hidden_dim)  ← 마지막 은닉 상태
  → FC (h_last)    → (batch, n_classes)
```

### LSTM의 게이트

| 게이트 | 역할 |
|--------|------|
| Forget gate | 이전 정보 중 무엇을 버릴지 |
| Input gate | 새 정보 중 무엇을 저장할지 |
| Output gate | 무엇을 출력할지 |

### RNN을 하나만 하는 이유

- 현대 NLP의 주력은 Transformer
- RNN/LSTM은 "시퀀스를 순차적으로 처리하는 모델"의 개념을 이해하는 데 의미가 있다
- 이후 Self-Attention이 왜 이를 대체했는지 체감할 수 있다

### CNN과의 비교

| | CNN | LSTM |
|---|---|---|
| 입력 | 고정 크기 (이미지) | 가변 길이 (시퀀스) |
| 처리 | 병렬 (합성곱) | 순차 (시간축) |
| 장점 | 공간 패턴 | 시간적 의존성 |
