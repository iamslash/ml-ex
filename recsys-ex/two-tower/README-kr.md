# Two-Tower Retrieval (투타워 검색)

> 유저와 아이템을 별도 타워로 임베딩하여 대규모 후보 검색을 수행하는 모델.

## 실행 방법

```bash
cd recsys-ex/two-tower
python two_tower.py
```

## 핵심 개념

### 구조

```
유저 특성 → User Tower → 유저 임베딩 (L2 정규화)
                                          → 내적 → 점수
아이템 특성 → Item Tower → 아이템 임베딩 (L2 정규화)
```

### In-batch Negative Sampling

배치 내 다른 아이템을 negative로 활용. 별도 negative 샘플링 불필요.

### MF와의 비교

| | MF | Two-Tower |
|---|---|---|
| 입력 | ID만 | 특성 벡터 |
| 콜드스타트 | 불가 | 특성으로 대응 |
| 검색 | 전체 계산 | ANN으로 밀리초 검색 |
