# GCN (Graph Convolutional Network, 그래프 합성곱 신경망)

> 정규화된 인접 행렬을 통해 이웃 노드의 특성을 집계하는 메시지 패싱 방식으로 노드 분류를 수행하는 그래프 신경망

## 실행 방법

```bash
cd graph-ex/gcn
python gcn.py
```

의존성: `torch`, `numpy`

3개 커뮤니티 각 10개 노드(총 30노드)의 합성 그래프에서 노드 분류를 수행한다. PyG 없이 순수 PyTorch로 구현되었다.

## 핵심 개념

### 그래프 합성곱 연산
각 노드가 이웃 노드의 특성을 집계하여 새 표현을 학습한다.

```
H^(l+1) = sigma(A_norm @ H^(l) @ W^(l))
```

- `A_norm`: 정규화된 인접 행렬
- `H^(l)`: l번째 레이어의 노드 특성 행렬
- `W^(l)`: 학습 가능한 가중치 행렬
- `sigma`: 활성화 함수 (ReLU)

구현에서는 `nn.Linear`가 `@ W` 역할을 수행한다.

```python
def forward(self, x, adj):
    return self.linear(adj @ x)  # A_norm @ X @ W
```

### Degree Normalization
자기 루프를 추가한 인접 행렬을 차수 행렬로 정규화한다.

```python
adj = adj + torch.eye(n_nodes)         # self-loop: 자기 자신도 집계
degree = adj.sum(dim=1)
D_inv_sqrt = torch.diag(1.0 / degree.sqrt())
adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # D^{-1/2} A D^{-1/2}
```

정규화 없이는 차수가 높은 노드의 특성이 집계 시 지배적이 된다. `D^{-1/2} A D^{-1/2}`는 행과 열 모두 정규화하여 대칭성을 유지한다.

### 모델 구조
```
노드 특성 (30, 8)
     |
GCN Layer 1: A_norm @ X @ W1  -> (30, 16)  + ReLU
     |
GCN Layer 2: A_norm @ X @ W2  -> (30, 3)
     |
Softmax -> 클래스 예측
```

2레이어 GCN은 2-hop 이웃까지의 정보를 집계한다.

### Transductive Learning
학습 시 전체 그래프 구조(A_norm)와 모든 노드 특성이 필요하다. 훈련/테스트 분할은 어느 노드의 레이블을 손실에 사용할지만 결정한다.

```python
loss = criterion(output[train_mask], labels[train_mask])  # 학습 노드 레이블만 사용
pred = model(node_features, adj_norm)  # 전체 그래프 입력
```

## 관련 모델과의 비교

| 항목 | GCN | GAT | GraphSAGE | GIN |
|------|-----|-----|-----------|-----|
| 집계 방식 | 동일 가중치 | 학습된 어텐션 | 샘플링 후 집계 | 합산 후 MLP |
| 이웃 가중치 | Degree 기반 고정 | 데이터 적응적 | 균일/최대 | 동일 |
| 확장성 | 작은 그래프 | 작은 그래프 | 대규모 가능 | 대규모 가능 |
| 이론적 표현력 | 중간 | 중간 | 중간 | WL 동치 (최대) |
| 귀납적 학습 | 불가 | 불가 | 가능 | 가능 |

### GCN vs MLP (그래프 구조 무시)
- **MLP**: 각 노드를 독립적으로 처리, 그래프 구조 정보 미활용
- **GCN**: 이웃 정보를 집계하여 연결된 노드끼리 비슷한 표현 학습

### Self-loop의 역할
Self-loop이 없으면 노드가 자신의 이전 특성을 집계에 포함하지 못한다. 특히 고립 노드(이웃 없음)는 업데이트가 되지 않는 문제 발생.

### 레이어 수와 Over-smoothing
레이어가 많아질수록 모든 노드의 표현이 동일해지는 Over-smoothing 현상이 발생한다. 실무에서는 2~3레이어가 일반적으로 최적이다.
