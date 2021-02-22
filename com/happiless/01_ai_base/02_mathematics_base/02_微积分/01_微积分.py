#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 01_微积分.py    
# @Contact : Happiless.zhang (Happiless.zhang@fatritech.com)
#
# @Modify Time      @Author    @Version    @Desciption
# ------------      -------    --------    -----------
# 2021/2/7 17:34   Happiless.zhang      1.0         None
import numpy as np
from IPython import display
import matplotlib.pyplot as plt


def solve_quadratic_min_by_calculus():
    """
    通过微积分求解二次方程最小值
    :return:
    """

    # 定义二次函数
    def f(x):
        return x ** 2

    # 定义导数
    def d(x):
        return 2 * x

    x = float(input("请输入x的初始值: "))

    learning_rate = 0.9

    # 数据集
    x_temp = np.linspace(-x, x, 1000)
    y_temp = x_temp ** 2

    while abs(x) > 0.0000001:
        display.clear_output(wait=True)
        plt.clf()
        plt.plot(x_temp, y_temp)
        plt.scatter(x, f(x), s=30, c='red')
        plt.show()
        print('此时的x值：', x)
        plt.pause(0.7)
        dy = d(x)
        x -= learning_rate * dy
    else:
        display.clear_output(wait=True)
        plt.clf()
        plt.plot(x_temp, y_temp)
        plt.scatter(x, f(x), s=30, c='red')
        plt.show()
        print("最终x的值为：", x)


def solve_regular_figure_area():
    pass


def solve_irregular_figure_area_approximately():
    pass


solve_quadratic_min_by_calculus()
