import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -----------------------------
# Generate Dataset
# -----------------------------
X, y = make_regression(
    n_samples=10000,
    n_features=10,
    noise=10,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add Bias Term
X = np.c_[np.ones((X.shape[0], 1)), X]

# -----------------------------
# Loss Function
# -----------------------------
def mse_loss(X, y, theta):
    pred = X @ theta
    return np.mean((pred - y) ** 2)

# -----------------------------
# Batch Gradient Descent
# -----------------------------
def batch_gd(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)

    updates = 0
    start = time.time()

    for _ in range(epochs):
        pred = X @ theta
        gradient = (2/m) * X.T @ (pred - y)

        theta -= lr * gradient
        updates += 1

    end = time.time()

    return theta, updates, end-start

# -----------------------------
# Stochastic Gradient Descent
# -----------------------------
def stochastic_gd(X, y, lr=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)

    updates = 0
    start = time.time()

    for _ in range(epochs):
        indices = np.random.permutation(m)

        for i in indices:
            xi = X[i:i+1]
            yi = y[i]

            pred = xi @ theta
            gradient = 2 * xi.T.flatten() * (pred - yi)

            theta -= lr * gradient
            updates += 1

    end = time.time()

    return theta, updates, end-start

# -----------------------------
# Mini-Batch Gradient Descent
# -----------------------------
def mini_batch_gd(X, y, batch_size=64, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)

    updates = 0
    start = time.time()

    for _ in range(epochs):
        indices = np.random.permutation(m)

        for start_idx in range(0, m, batch_size):
            batch_idx = indices[start_idx:start_idx+batch_size]

            Xb = X[batch_idx]
            yb = y[batch_idx]

            pred = Xb @ theta
            gradient = (2/len(yb)) * Xb.T @ (pred - yb)

            theta -= lr * gradient
            updates += 1

    end = time.time()

    return theta, updates, end-start

# -----------------------------
# Run Experiments
# -----------------------------
results = []

# Batch GD
theta_b, upd_b, time_b = batch_gd(X, y)
loss_b = mse_loss(X, y, theta_b)

results.append([
    "Batch GD",
    upd_b,
    round(time_b, 4),
    round(loss_b, 4)
])

# SGD
theta_s, upd_s, time_s = stochastic_gd(X, y)
loss_s = mse_loss(X, y, theta_s)

results.append([
    "SGD",
    upd_s,
    round(time_s, 4),
    round(loss_s, 4)
])

# Mini Batch GD
theta_m, upd_m, time_m = mini_batch_gd(X, y, batch_size=64)
loss_m = mse_loss(X, y, theta_m)

results.append([
    "Mini-Batch GD (64)",
    upd_m,
    round(time_m, 4),
    round(loss_m, 4)
])

# -----------------------------
# Display Results
# -----------------------------
print("\nComparison of Gradient Descent Variants\n")

print("{:<20} {:<20} {:<15} {:<15}".format(
    "Method",
    "Parameter Updates",
    "Time(s)",
    "Final Loss"
))

print("-"*70)

for row in results:
    print("{:<20} {:<20} {:<15} {:<15}".format(*row))
