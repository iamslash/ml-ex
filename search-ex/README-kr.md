# 검색 (Search & Retrieval)

> 정보 검색의 기초부터 신경망 기반 의미 검색까지.

## 학습 순서

| # | 모델 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | BM25 | [bm25](bm25/) | TF-IDF, 역색인, 스코어링 |
| 2 | Semantic Search | [semantic-search](semantic-search/) | 문장 임베딩, cosine similarity, contrastive learning |
| 3 | Re-ranker | [reranker](reranker/) | cross-encoder, 정밀 랭킹 |

## 검색 파이프라인

```
쿼리 → 1차 검색 (BM25/ANN) → 후보 N개 → Re-ranker → 최종 K개
          빠르지만 거침           정밀하지만 느림
```
