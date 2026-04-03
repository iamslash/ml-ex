"""
Minimal CNN from scratch.
순수 numpy로 구현한 합성곱 신경망. MNIST 유사 데이터로 숫자 분류.
"""

import numpy as np

np.random.seed(42)

# --- 합성 MNIST 유사 데이터 생성 ---
# 실제 MNIST 대신 5x5 패턴으로 3클래스 분류
def make_patterns(n_per_class=100):
    """3가지 패턴의 5x5 이미지 생성"""
    X, y = [], []
    for _ in range(n_per_class):
        # 클래스 0: 수평선 패턴
        img = np.zeros((5, 5))
        img[2, :] = 1
        img += np.random.randn(5, 5) * 0.2
        X.append(img); y.append(0)

        # 클래스 1: 수직선 패턴
        img = np.zeros((5, 5))
        img[:, 2] = 1
        img += np.random.randn(5, 5) * 0.2
        X.append(img); y.append(1)

        # 클래스 2: 대각선 패턴
        img = np.zeros((5, 5))
        for i in range(5): img[i, i] = 1
        img += np.random.randn(5, 5) * 0.2
        X.append(img); y.append(2)

    X = np.array(X)  # (300, 5, 5)
    y = np.array(y)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

X, y = make_patterns(100)
X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

print(f"train: {len(X_train)}, test: {len(X_test)}")
print(f"image shape: {X_train[0].shape}, classes: {len(np.unique(y))}")

# --- Conv2D (단일 필터, valid padding) ---
def conv2d(image, kernel):
    """image: (H, W), kernel: (kH, kW) -> output: (H-kH+1, W-kW+1)"""
    H, W = image.shape
    kH, kW = kernel.shape
    oH, oW = H - kH + 1, W - kW + 1
    output = np.zeros((oH, oW))
    for i in range(oH):
        for j in range(oW):
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return output

# --- Max Pooling ---
def max_pool2d(image, size=2):
    """image: (H, W) -> output: (H//size, W//size)"""
    H, W = image.shape
    oH, oW = H // size, W // size
    output = np.zeros((oH, oW))
    for i in range(oH):
        for j in range(oW):
            output[i, j] = np.max(image[i*size:(i+1)*size, j*size:(j+1)*size])
    return output

# --- ReLU ---
def relu(x):
    return np.maximum(0, x)

# --- Softmax ---
def softmax(z):
    z_shift = z - z.max()
    exp_z = np.exp(z_shift)
    return exp_z / exp_z.sum()

# --- 모델 파라미터 ---
n_filters = 4
kernel_size = 3
n_classes = 3

# Conv 필터: (n_filters, 3, 3)
filters = np.random.randn(n_filters, kernel_size, kernel_size) * 0.3

# Conv 출력: 5x5 -> 3x3 (valid) -> max_pool -> 1x1 -> flatten: n_filters
# FC: (n_filters, n_classes)
W_fc = np.random.randn(n_filters, n_classes) * 0.3
b_fc = np.zeros(n_classes)

print(f"filters: {n_filters}x{kernel_size}x{kernel_size}")
print(f"params: {n_filters*kernel_size*kernel_size + n_filters*n_classes + n_classes}")

# --- Forward ---
def forward(image):
    # Conv + ReLU
    conv_outputs = []
    for f in range(n_filters):
        out = conv2d(image, filters[f])  # (3, 3)
        out = relu(out)
        conv_outputs.append(out)

    # Max Pool -> flatten
    pooled = []
    for out in conv_outputs:
        p = max_pool2d(out, size=3)  # (3,3) -> (1,1)
        pooled.append(p[0, 0])
    pooled = np.array(pooled)  # (n_filters,)

    # FC + Softmax
    logits = pooled @ W_fc + b_fc  # (n_classes,)
    probs = softmax(logits)

    return probs, pooled, conv_outputs

# --- 학습 (수치 미분) ---
learning_rate = 0.01
num_epochs = 30

def compute_loss(probs, label):
    return -np.log(probs[label] + 1e-7)

def numerical_gradient(param, image, label, eps=1e-4):
    """수치 미분으로 그래디언트 계산"""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        param[idx] = old_val + eps
        probs_plus, _, _ = forward(image)
        loss_plus = compute_loss(probs_plus, label)

        param[idx] = old_val - eps
        probs_minus, _, _ = forward(image)
        loss_minus = compute_loss(probs_minus, label)

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = old_val
        it.iternext()
    return grad

# Mini-batch 학습 (수치 미분은 느리므로 소규모로)
batch_size = 10

for epoch in range(num_epochs):
    indices = np.random.permutation(len(X_train))[:batch_size]
    epoch_loss = 0

    for idx in indices:
        image, label = X_train[idx], y_train[idx]
        probs, _, _ = forward(image)
        epoch_loss += compute_loss(probs, label)

        # 수치 미분으로 그래디언트 계산
        grad_filters = numerical_gradient(filters, image, label)
        grad_W_fc = numerical_gradient(W_fc, image, label)
        grad_b_fc = numerical_gradient(b_fc, image, label)

        # 파라미터 업데이트
        filters -= learning_rate * grad_filters
        W_fc -= learning_rate * grad_W_fc
        b_fc -= learning_rate * grad_b_fc

    avg_loss = epoch_loss / batch_size

    if epoch % 5 == 0:
        # 테스트 정확도
        correct = 0
        for i in range(len(X_test)):
            probs, _, _ = forward(X_test[i])
            if np.argmax(probs) == y_test[i]:
                correct += 1
        acc = correct / len(X_test) * 100
        print(f"epoch {epoch:2d} | loss {avg_loss:.4f} | test acc {acc:.1f}%")

# --- 최종 평가 ---
correct = 0
for i in range(len(X_test)):
    probs, _, _ = forward(X_test[i])
    if np.argmax(probs) == y_test[i]:
        correct += 1
print(f"\nfinal test accuracy: {correct/len(X_test)*100:.1f}%")
