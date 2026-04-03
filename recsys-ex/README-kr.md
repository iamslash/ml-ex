# 추천 시스템 (Recommendation System)

> 임베딩, 검색, 랭킹의 핵심을 학습하는 추천 시스템 실습.

## 학습 순서

| # | 모델 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | Matrix Factorization | [matrix-factorization](matrix-factorization/) | 임베딩 내적, SGD, bias, 정규화 |
| 2 | Two-Tower | [two-tower](two-tower/) | 쿼리/아이템 타워, in-batch negative, retrieval |
| 3 | NCF | [ncf](ncf/) | GMF + MLP, implicit feedback, ranking |

## 추천 파이프라인

```
후보 생성 (Retrieval) → 랭킹 (Ranking) → 필터링/재랭킹
     Two-Tower              NCF
     Matrix Factorization
```
