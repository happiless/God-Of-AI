#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 02_homework_gradient_descent.py
# @todo    : 梯度下降
import math

import numpy as np
import matplotlib.pyplot as plt
import random
from IPython import display


def func(x):
    return 10 * x ** 2 + 37 * x + 9


def gradient(x):
    return 20 * x + 37


def loss(y, y_hat):
    return math.fabs(y - y_hat)


if __name__ == '__main__':
    x = np.linspace(-10, 10)
    x_optimal = random.choice(x)
    plt.plot(x, func(x))
    alpha = 1e-3
    e = 1
    fig = plt.figure()
    ax = plt.gca()
    while e > 0.001:
        display.clear_output(wait=True)
        plt.clf()
        y = func(x_optimal)
        x_optimal -= alpha * gradient(x_optimal)
        y_hat = func(x_optimal)
        e = loss(y, y_hat)
        plt.plot(x, func(x))
        plt.scatter(x_optimal, func(x_optimal), color='red')
        plt.show()
        plt.pause(0.7)
    else:
        display.clear_output(wait=True)
        plt.clf()
        plt.plot(x, func(x))
        plt.scatter(x_optimal, func(x_optimal), s=30, c='red')
        plt.show()
        print("最终x的值为：", x)
