# Item2Vec (아이템 임베딩)

> Word2Vec의 Skip-gram을 행동 시퀀스에 적용하여 아이템 임베딩을 학습.

## 실행 방법

```bash
cd recsys-ex/item2vec
python item2vec.py
```

## 핵심 개념

### 아이디어

NLP에서 "단어의 의미는 주변 단어로 정의된다"처럼, 추천에서 "아이템의 특성은 함께 소비된 아이템으로 정의된다".

```
유저 행동 시퀀스 = "문장"
각 아이템 = "단어"
Skip-gram으로 아이템 임베딩 학습
```

### Negative Sampling

```
positive: log σ(v_center · v_context)
negative: Σ log σ(-v_center · v_neg)
```

빈도의 0.75승으로 negative 분포를 smoothing하여 희귀 아이템도 골고루 샘플링.

### 유저 임베딩 (Cold-start)

- 평균: 유저가 소비한 아이템 임베딩의 평균
- 최근: 최근 k개 아이템 임베딩의 평균
- 신규 유저: 인기 아이템 임베딩의 평균
