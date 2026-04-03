# FT-Transformer (피처 토크나이저 + 트랜스포머)

> 수치형 및 범주형 피처를 각각 토큰으로 변환한 뒤 Transformer Self-Attention으로 피처 간 상호작용을 학습하는 테이블 데이터 모델

## 실행 방법

```bash
cd recsys-ex/ft-transformer
python ft_transformer.py
```

의존성: `torch`

출력 예시:
- 에폭별 train loss와 test 정확도 (10 에폭마다)

## 핵심 개념

### Feature Tokenizer
각 피처를 동일한 차원(`d_token`)의 토큰으로 변환:

```python
# 수치형: scalar를 d_token 차원으로 projection
num_token = Linear(1, d_token)(x[:, i:i+1])   # (batch, d_token)

# 범주형: embedding lookup
cat_token = Embedding(n_vals, d_token)(x_cat)  # (batch, d_token)
```

- 수치형 피처마다 별도의 Linear가 있어 각 피처가 독립적인 의미 공간으로 임베딩됨
- 범주형 피처는 ID 임베딩 테이블 사용

### [CLS] 토큰

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
tokens = cat([cls, feature_tokens], dim=1)
# -> (batch, 1+n_features, d_token)
```

BERT의 [CLS] 토큰과 동일한 역할:
- Transformer 통과 후 [CLS] 위치의 출력만 사용하여 최종 예측

### Transformer Self-Attention

```
입력: (batch, 1+12, 32)   # 1 CLS + 8 수치형 + 4 범주형
       |
Self-Attention           # 모든 피처 쌍의 상호작용 학습
Feed-Forward             # 각 위치별 변환
       |
출력: (batch, 1+12, 32)
       |
cls_out = out[:, 0, :]   # (batch, 32)
       |
fc_out -> 예측값
```

Self-Attention은 각 피처가 다른 피처에 얼마나 주목할지를 학습:
- "피처 A와 B의 교차항이 중요할 때 높은 attention weight"

### 전체 구조 요약

```
x_num (8차원)   x_cat (4개 범주형)
      |                |
  수치형 토크나이저   범주형 토크나이저
  (Linear x 8)     (Embedding x 4)
      |                |
      +---- stack ------+
             |
         + [CLS] prepend
             |
     (batch, 13, 32)
             |
    TransformerEncoder (2 layers, 4 heads)
             |
     (batch, 13, 32)
             |
       cls 위치[0] 추출
             |
       Linear -> 예측
```

## 관련 모델과의 비교

| 모델 | 수치형 처리 | 범주형 처리 | 피처 간 상호작용 | 파라미터 수 |
|------|------------|------------|-----------------|------------|
| **FT-Transformer** | Linear 토크나이저 | Embedding | Self-Attention | 많음 |
| **DCN-v2** | 직접 입력 | Embedding 후 concat | Cross Layer | 중간 |
| **TabNet** | 직접 입력 | Embedding | Attention mask | 중간 |
| **GBDT (XGBoost)** | 직접 입력 | 인코딩 필요 | 트리 분기 | N/A |
| **MLP** | 직접 입력 | Embedding 후 concat | 없음 | 적음 |

### FT-Transformer vs DCN-v2
- DCN-v2: Cross Layer로 명시적 교차항 생성, 빠름
- FT-Transformer: Self-Attention으로 암묵적 교차 학습, 표현력 높음
- 피처 수가 많을수록 FT-Transformer의 O(n^2) attention 비용이 증가

### FT-Transformer vs TabNet
- TabNet: Sequential attention으로 피처 선택 (interpretable)
- FT-Transformer: 모든 피처를 동시에 attend (더 높은 표현력)
- TabNet이 해석 가능성 면에서 유리

### FT-Transformer vs GBDT
- GBDT (XGBoost, LightGBM): 테이블 데이터에서 여전히 강력한 베이스라인
- FT-Transformer: 대규모 데이터와 피처가 많을 때 강점
- 소규모 테이블 데이터에서는 GBDT가 더 강한 경우 많음

### 추천 시스템에서의 위치
- 검색(Retrieval): Two-Tower (빠른 인코딩 필요)
- 랭킹(Ranking): DCN-v2, FT-Transformer (정밀도 중요)
- FT-Transformer는 피처가 다양하고 상호작용이 복잡한 랭킹에 적합

### 참고 논문
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
