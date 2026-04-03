# Transformer (인코더-디코더)

> 어텐션만으로 구성된 seq2seq 모델. 시퀀스 반전 과제로 구조를 학습한다.

## 실행 방법

```bash
cd pytorch-ex/transformer
python transformer.py
```

## 핵심 개념

### 구조

```
입력 시퀀스 → Encoder → 컨텍스트 표현
                           ↓
출력 시퀀스 → Decoder → 다음 토큰 예측
```

### Encoder Layer

```
x → Multi-Head Self-Attention → Add & LayerNorm → FFN → Add & LayerNorm
```

- 입력 시퀀스의 모든 위치가 서로를 참조 (양방향)

### Decoder Layer

```
x → Masked Self-Attention → Add & Norm
  → Cross-Attention (Encoder 출력 참조) → Add & Norm
  → FFN → Add & Norm
```

- Masked Self-Attention: 미래 토큰을 볼 수 없음 (causal)
- Cross-Attention: 디코더가 인코더 출력을 참조하는 핵심 메커니즘

### Positional Encoding

Transformer에는 순서 개념이 없으므로, sin/cos 함수로 위치 정보를 주입한다.

### Teacher Forcing vs Greedy Decoding

| | Teacher Forcing (학습) | Greedy Decoding (추론) |
|---|---|---|
| 입력 | 정답 시퀀스 | 이전 스텝의 예측값 |
| 속도 | 빠름 (병렬) | 느림 (순차) |
| 용도 | 학습 | 추론 |

## 과제: 시퀀스 반전

```
입력: [BOS, 3, 7, 2, 5, EOS]
출력: [BOS, 5, 2, 7, 3, EOS]
```

단순하지만 Encoder가 전체 시퀀스를 이해하고, Decoder가 역순으로 생성해야 하므로 Transformer의 모든 구성요소를 사용한다.
