# -----------------------------------------------------------------------------
# Implementation of a single neuron (logistic regression) for binary classification.
#
# This script demonstrates:
# - Building a simple neural unit from scratch using only NumPy.
# - Training a linear classifier using binary cross-entropy loss and gradient descent.
# - Generating synthetic, linearly separable data for testing the model.
# - Reporting loss during training to monitor convergence.
#
# No external machine learning libraries are used; this is fully educational and
# suitable for understanding the fundamentals of neural network training.
# 
# Author: Yarden Carmel
# -----------------------------------------------------------------------------



import numpy as np


def binary_cross_entropy(y, y_hat):
    N = len(y)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1-eps)
    loss_sum = -(np.dot(y, np.log(y_hat)) + np.dot((1-y), np.log(1-y_hat)))
    loss = loss_sum/N
    return loss


def loss_and_grads(X, y, w, b):
    N = len(y)
    z = X @ w + b
    y_hat = 1.0 / (1.0 + np.exp(-z))
    loss = binary_cross_entropy(y, y_hat)
    error = y_hat-y
    dldw = (error @ X) / N
    dldb = error.sum() / N
    return loss, dldw, dldb


def train(X, y, lr=0.1, num_epochs=1000):
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for epoch in range(1, num_epochs+1):
        loss, dldw, dldb = loss_and_grads(X, y, w, b)
        w = w - lr * dldw
        b = b - lr * dldb
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | loss = {loss:.4f}")

    return w, b


if __name__ == "__main__":
    N=200
    d=2
    N_per_class = N // 2

    X0 = np.random.randn(N_per_class, d) + np.array([-1.0, -1.0])
    y0 = np.zeros(N_per_class)

    X1 = np.random.randn(N_per_class, d) + np.array([1.0, 1.0])
    y1 = np.ones(N_per_class)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    w_learned, b_learned = train(X, y, lr=0.1, num_epochs=1000)

