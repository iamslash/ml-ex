# 추천 시스템 (Recommendation System)

> 임베딩, 검색, 랭킹의 핵심을 학습하는 추천 시스템 실습.

## 학습 순서

| # | 모델 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | Matrix Factorization | [matrix-factorization](matrix-factorization/) | 임베딩 내적, SGD, bias, 정규화 |
| 2 | Two-Tower | [two-tower](two-tower/) | 쿼리/아이템 타워, in-batch negative, retrieval |
| 3 | NCF | [ncf](ncf/) | GMF + MLP, implicit feedback, ranking |
| 4 | DCN-v2 | [dcn](dcn/) | Deep & Cross Network, 명시적 feature interaction |
| 5 | PLE Multi-Task | [ple-mtl](ple-mtl/) | expert 네트워크, SwiGLU, uncertainty loss |
| 6 | Item2Vec | [item2vec](item2vec/) | 행동 시퀀스 → 임베딩, negative sampling |
| 7 | FT-Transformer | [ft-transformer](ft-transformer/) | 테이블 데이터 + Transformer, feature tokenizer |
| 8 | Offline RL (CQL) | [offline-rl](offline-rl/) | DQN + Conservative Q-Learning, 오프라인 정책 학습 |

## 추천 파이프라인

```
임베딩 학습 (Item2Vec, MF)
  → 후보 생성 (Two-Tower)
  → 랭킹 (NCF, DCN, FT-Transformer, PLE-MTL)
  → 정책 최적화 (Offline RL)
```
