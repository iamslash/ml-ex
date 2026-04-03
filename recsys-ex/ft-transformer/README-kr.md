# FT-Transformer (Feature Tokenizer + Transformer)

> 테이블 데이터의 각 특성을 토큰으로 변환하여 Transformer에 입력하는 모델.

## 실행 방법

```bash
cd recsys-ex/ft-transformer
python ft_transformer.py
```

## 핵심 개념

### Feature Tokenizer

```
수치형 특성 → Linear projection → 토큰
범주형 특성 → Embedding → 토큰
[CLS] 토큰 추가 → Transformer 입력
```

각 특성을 동일 차원의 토큰으로 변환하여 Transformer가 특성 간 관계를 어텐션으로 학습.

### 구조

```
특성들 → Feature Tokenizer → [CLS, tok1, tok2, ...] → Transformer Encoder → [CLS] → FC → 예측
```

### XGBoost/LightGBM과의 비교

| | GBDT | FT-Transformer |
|---|---|---|
| 특성 교차 | 트리 분할로 자동 | 어텐션으로 자동 |
| 범주형 처리 | 인코딩 필요 | 임베딩으로 직접 |
| 대규모 데이터 | 빠름 | GPU 필요 |
| 성능 | 테이블 데이터 강자 | GBDT와 경쟁 가능 |
