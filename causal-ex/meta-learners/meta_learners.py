"""
Meta-Learners for Causal Inference.
S/T/X-Learner 비교로 처리 효과(CATE) 추정.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)

# --- 합성 인과 데이터 ---
n = 2000
X = np.random.randn(n, 5)

# 처리 배정 (propensity는 X[:, 0]에 의존)
propensity = 1 / (1 + np.exp(-X[:, 0]))
T = (np.random.rand(n) < propensity).astype(int)

# 잠재 결과
# Y(0) = X[:, 1] + noise
# Y(1) = X[:, 1] + tau(X) + noise
# tau(X) = 2 * (X[:, 2] > 0) + X[:, 3]  (heterogeneous treatment effect)
tau_true = 2 * (X[:, 2] > 0).astype(float) + X[:, 3]
Y0 = X[:, 1] + np.random.randn(n) * 0.5
Y1 = Y0 + tau_true
Y = T * Y1 + (1 - T) * Y0

X_train, X_test, T_train, T_test, Y_train, Y_test, tau_train, tau_test = \
    train_test_split(X, T, Y, tau_true, test_size=0.3, random_state=42)

print(f"train: {len(X_train)}, test: {len(X_test)}")
print(f"treatment rate: {T.mean():.2f}")
print(f"true ATE: {tau_true.mean():.2f}")

# --- S-Learner ---
print("\n=== S-Learner ===")
# 처리 T를 특성으로 포함하여 단일 모델 학습
X_with_T_train = np.column_stack([X_train, T_train])

s_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
s_model.fit(X_with_T_train, Y_train)

# CATE = E[Y|X, T=1] - E[Y|X, T=0]
X_test_t1 = np.column_stack([X_test, np.ones(len(X_test))])
X_test_t0 = np.column_stack([X_test, np.zeros(len(X_test))])
cate_s = s_model.predict(X_test_t1) - s_model.predict(X_test_t0)

rmse_s = np.sqrt(np.mean((cate_s - tau_test) ** 2))
print(f"  ATE estimate: {cate_s.mean():.2f}")
print(f"  CATE RMSE: {rmse_s:.4f}")

# --- T-Learner ---
print("\n=== T-Learner ===")
# 처리/대조 각각 별도 모델
t1_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
t0_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

t1_model.fit(X_train[T_train == 1], Y_train[T_train == 1])
t0_model.fit(X_train[T_train == 0], Y_train[T_train == 0])

cate_t = t1_model.predict(X_test) - t0_model.predict(X_test)

rmse_t = np.sqrt(np.mean((cate_t - tau_test) ** 2))
print(f"  ATE estimate: {cate_t.mean():.2f}")
print(f"  CATE RMSE: {rmse_t:.4f}")

# --- X-Learner ---
print("\n=== X-Learner ===")
# Step 1: T-Learner와 동일하게 mu_0, mu_1 학습
mu1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
mu0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
mu1.fit(X_train[T_train == 1], Y_train[T_train == 1])
mu0.fit(X_train[T_train == 0], Y_train[T_train == 0])

# Step 2: 잔차 (imputed treatment effect)
# 처리군: D_1 = Y_1 - mu_0(X)
D1 = Y_train[T_train == 1] - mu0.predict(X_train[T_train == 1])
# 대조군: D_0 = mu_1(X) - Y_0
D0 = mu1.predict(X_train[T_train == 0]) - Y_train[T_train == 0]

# Step 3: 잔차를 예측하는 모델
tau1_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
tau0_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
tau1_model.fit(X_train[T_train == 1], D1)
tau0_model.fit(X_train[T_train == 0], D0)

# Step 4: Propensity-weighted 결합
prop_model = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
prop_model.fit(X_train, T_train)
e_x = prop_model.predict_proba(X_test)[:, 1]

cate_x = e_x * tau1_model.predict(X_test) + (1 - e_x) * tau0_model.predict(X_test)

rmse_x = np.sqrt(np.mean((cate_x - tau_test) ** 2))
print(f"  ATE estimate: {cate_x.mean():.2f}")
print(f"  CATE RMSE: {rmse_x:.4f}")

# --- 비교 ---
print("\n=== Comparison ===")
print(f"  {'Method':<12s} | {'ATE':>6s} | {'CATE RMSE':>10s}")
print(f"  {'S-Learner':<12s} | {cate_s.mean():>6.2f} | {rmse_s:>10.4f}")
print(f"  {'T-Learner':<12s} | {cate_t.mean():>6.2f} | {rmse_t:>10.4f}")
print(f"  {'X-Learner':<12s} | {cate_x.mean():>6.2f} | {rmse_x:>10.4f}")
print(f"  {'True':<12s} | {tau_test.mean():>6.2f} | {'0.0000':>10s}")

# --- 서브그룹 분석 ---
print("\n=== Subgroup Analysis (X-Learner) ===")
for feat_idx, name in [(2, "X2>0"), (2, "X2<=0")]:
    if ">" in name:
        mask = X_test[:, feat_idx] > 0
    else:
        mask = X_test[:, feat_idx] <= 0
    print(f"  {name}: estimated CATE={cate_x[mask].mean():.2f}, true={tau_test[mask].mean():.2f}")
