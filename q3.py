#!/usr/bin/env python3
# RNN Q3 forward/backward with sigmoid activations (hidden + output) and BPTT
# All inputs are embedded below. Prints results rounded to 4 decimals.

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Inputs
U = np.array([
    [-1.1106, -0.0596, -0.4091, -0.0648],
    [-0.6809, -0.6610, -1.2819,  1.0004],
    [ 0.0142,  0.3060, -0.2849, -0.8013]
], dtype=float)

V = np.array([[ 1.1086],
                [-0.4641],
                [ 0.6646],
                [-0.5301]], dtype=float)

W = np.array([[ 0.0918, -0.9585,  0.8270, -0.2248],
                [ 0.2738, -0.3263, -0.1596, -1.3147],
                [-2.1335,  2.4787, -0.8577, -0.0218],
                [ 0.4040,  1.5391,  0.3984, -0.8801]], dtype=float)

c = np.array([[0.8004]], dtype=float)  # output bias
b = np.array([[ 0.0409],
                [ 0.1551],
                [-0.3211],
                [ 2.1458]], dtype=float)  # hidden bias

X = np.array([
    [-2.5092,  1.9732, -8.8383,  4.1615,  6.6489],
    [ 9.0143, -6.8796,  7.3235, -9.5883, -5.7532],
    [ 4.6399, -6.8801,  2.0223,  9.3982, -6.3635]
], dtype=float)  # (3 x T)

d = np.array([[ 1.1963,  2.0975,  0.3877,  1.0453, -1.3718]], dtype=float)  # (1 x T)

T = X.shape[1]
H = W.shape[0]

# ----- Forward pass -----
h = np.zeros((H, T+1))  # h^0 = 0
a_h = np.zeros((H, T))
a_y = np.zeros((1, T))
y   = np.zeros((1, T))

for t in range(T):
    x_t = X[:, [t]]                     # (3x1)
    a_h[:, [t]] = U.T @ x_t + W.T @ h[:, [t]] + b  # (4x1)
    h[:, [t+1]] = sigmoid(a_h[:, [t]])
    a_y[:, [t]] = V.T @ h[:, [t+1]] + c            # (1x1)
    y[:, [t]]   = sigmoid(a_y[:, [t]])

# ----- Backward pass (BPTT) -----
# loss: 0.5 * sum_t (y_t - d_t)^2  -> dL/da_y = (y - d) * sigma'(a_y)
delta_y = (y - d) * (y * (1 - y))  # (1xT)
delta_h = np.zeros((H, T))

for t in reversed(range(T)):
    term = (V @ delta_y[:, [t]])  # (4x1)
    if t < T-1:
        term += W @ delta_h[:, [t+1]]
    delta_h[:, [t]] = term * (h[:, [t+1]] * (1 - h[:, [t+1]]))

# ----- Gradients -----
grad_V = h[:, 1:] @ delta_y.T                   # (4x1)
grad_c = delta_y.sum(axis=1, keepdims=True)     # (1x1)
grad_b = delta_h.sum(axis=1, keepdims=True)     # (4x1)

grad_U = np.zeros_like(U)
grad_W = np.zeros_like(W)
for t in range(T):
    x_t   = X[:, [t]]
    h_prev = h[:, [t]]
    dht   = delta_h[:, [t]]
    grad_U += x_t @ dht.T       # (3x4)
    grad_W += h_prev @ dht.T    # (4x4)

# ----- Pretty print (rounded to 4 decimals) -----
np.set_printoptions(precision=4, suppress=True)

print("=== 1) Output Sequence y(t) ===")
print(y.flatten().round(4))

print("\n=== 2) Deltas ===")
print("delta_y(t):")
print(delta_y.flatten().round(4))

print("\nHidden deltas delta_h (rows=h1..h4, cols=t1..t5):")
print(delta_h.round(4))

print("\n=== 3) Gradients ===")
print("∇V:")
print(grad_V.round(4))
print("∇c:")
print(grad_c.round(4))
print("∇b:")
print(grad_b.round(4))
print("∇U:")
print(grad_U.round(4))
print("∇W:")
print(grad_W.round(4))
