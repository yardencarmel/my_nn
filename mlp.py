import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def forward_hidden(X, W1, b1):
    # compute z1 and a1
    z1 = X@W1 + b1
    a1 = relu(z1)
    return a1, z1


def forward_2layer(X, W1, b1, W2, b2):
    a1, z1 = forward_hidden(X, W1, b1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    return y_hat, (z1, a1, z2)