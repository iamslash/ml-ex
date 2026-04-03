# GAT (Graph Attention Network)

> 이웃 노드마다 어텐션 가중치를 학습하여 중요한 이웃에 더 집중하는 그래프 신경망.

## 실행 방법

```bash
cd graph-ex/gat
python gat.py
```

## 핵심 개념

### 그래프 어텐션

```
e_ij = LeakyReLU(a_left · h_i + a_right · h_j)
α_ij = softmax_j(e_ij)
h_i' = Σ α_ij · h_j
```

- 각 이웃 j에 대해 어텐션 점수 계산
- softmax로 정규화하여 가중 합산
- 중요한 이웃에 더 집중

### Multi-Head Attention

여러 어텐션 헤드를 병렬로 수행하여 다양한 관계 패턴을 포착. Transformer의 multi-head와 동일 개념.
