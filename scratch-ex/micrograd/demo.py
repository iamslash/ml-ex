"""
micrograd demo: binary classification on the make_moons dataset.
Based on karpathy/micrograd demo.ipynb
Requires: numpy, scikit-learn (for dataset generation)
"""

import random
import numpy as np
from engine import Value
from nn import MLP

# --- seed ---
np.random.seed(1337)
random.seed(1337)

# --- dataset ---
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # make y be -1 or 1
print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features")

# --- model ---
model = MLP(2, [16, 16, 1])  # 2-layer neural network
print(model)
print(f"number of parameters: {len(model.parameters())}")

# --- loss function ---
def loss(batch_size=None):
    # inline DataLoader
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

# --- training ---
for k in range(100):
    # forward
    total_loss, acc = loss()

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k % 10 == 0:
        print(f"step {k:3d} | loss {total_loss.data:.4f} | accuracy {acc*100:.1f}%")

# --- final result ---
total_loss, acc = loss()
print(f"\nfinal loss: {total_loss.data:.4f}, accuracy: {acc*100:.1f}%")
