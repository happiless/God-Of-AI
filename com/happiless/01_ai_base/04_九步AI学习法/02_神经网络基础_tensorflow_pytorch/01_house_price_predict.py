#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 5])

plt.scatter(x, y)
plt.grid()


def MAE_loss(y, y_hat):
    return np.mean(np.abs(y_hat - y))


def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))


def linear(x, k, b):
    return k * x + b


min_loss = float('inf')
best_k, best_b = 0, 0
for k in np.arange(-2, 2, 0.1):
    for b in np.arange(-10, 10, 0.1):
        y_hat = [linear(xi, k, b) for xi in list(x)]
        current_loss = MSE_loss(y, y_hat)
        if current_loss < min_loss:
            min_loss = current_loss
            best_k, best_b = k, b
            print('best k is {}, best b is {}'.format(best_k, best_b))

y_hat = linear(x, best_k, best_b)

plt.plot(x, y_hat, color='red')
plt.grid()
plt.show()
