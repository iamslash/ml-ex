# DCN-v2 (Deep & Cross Network)

> Cross Layer로 명시적 feature interaction을 자동 학습하는 CTR 예측 모델.

## 실행 방법

```bash
cd recsys-ex/dcn
python dcn.py
```

## 핵심 개념

### Cross Layer

```
x_{l+1} = x_0 * (W @ x_l + b) + x_l
```

- `x_0`과의 곱으로 원본 특성과의 교차항을 명시적으로 생성
- L개 레이어 = L+1차 교차항까지 자동 학습
- 수동 feature engineering 불필요

### DCN-v2 구조

```
입력 → Cross Network (명시적 교차) ─┐
                                     ├→ concat → FC → 예측
입력 → Deep Network (비선형 학습) ──┘
```

### MLP와의 비교

MLP는 교차항을 암묵적으로만 학습. DCN은 Cross Layer로 명시적 교차를 보장하면서 Deep Network로 비선형도 포착.
