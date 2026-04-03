"""
BM25 from scratch.
순수 Python으로 구현한 정보 검색 알고리즘.
"""

import math
from collections import Counter

# --- 문서 데이터 ---
documents = [
    "the cat sat on the mat",
    "the dog played in the park",
    "a cat and a dog are friends",
    "the quick brown fox jumps over the lazy dog",
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing deals with text and speech",
    "reinforcement learning trains agents with rewards",
    "convolutional neural networks are used for image recognition",
    "transformers revolutionized natural language processing",
    "recommendation systems predict user preferences",
    "search engines use inverted indexes for fast retrieval",
]

print(f"documents: {len(documents)}")

# --- 토큰화 ---
def tokenize(text):
    return text.lower().split()

# --- 역색인 구축 ---
doc_tokens = [tokenize(doc) for doc in documents]
doc_lengths = [len(tokens) for tokens in doc_tokens]
avg_dl = sum(doc_lengths) / len(doc_lengths)
N = len(documents)

# df: 각 단어가 몇 개 문서에 등장하는지
df = Counter()
for tokens in doc_tokens:
    for term in set(tokens):
        df[term] += 1

print(f"vocabulary: {len(df)} terms")
print(f"avg doc length: {avg_dl:.1f}")

# --- BM25 스코어 ---
def bm25_score(query, doc_idx, k1=1.5, b=0.75):
    """
    BM25(q, d) = Σ IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    """
    query_terms = tokenize(query)
    score = 0.0
    dl = doc_lengths[doc_idx]
    tf_doc = Counter(doc_tokens[doc_idx])

    for term in query_terms:
        if term not in tf_doc:
            continue

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        n = df.get(term, 0)
        idf = math.log((N - n + 0.5) / (n + 0.5) + 1)

        # TF normalization
        tf = tf_doc[term]
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))

        score += idf * tf_norm

    return score

def search(query, top_k=5):
    """쿼리에 대해 BM25 스코어가 높은 문서 반환"""
    scores = [(i, bm25_score(query, i)) for i in range(N)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(i, s, documents[i]) for i, s in scores[:top_k] if s > 0]

# --- 검색 테스트 ---
queries = [
    "cat and dog",
    "neural networks deep learning",
    "natural language processing",
    "search retrieval",
    "machine learning intelligence",
]

for query in queries:
    print(f"\nquery: \"{query}\"")
    results = search(query, top_k=3)
    if not results:
        print("  (no results)")
    for rank, (doc_id, score, doc) in enumerate(results, 1):
        print(f"  {rank}. [{score:.3f}] {doc}")
