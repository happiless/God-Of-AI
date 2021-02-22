#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 02_numpy常用函数.py    
# numpy中统计函数的使用
import numpy as np


# 最大、最小值
def work1():
    print("最大、最小值")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(np.min(a))
    print(np.min(a, 0))
    print(np.min(a, 1))
    print(np.max(a))
    print(np.max(a, 0))
    print(np.max(a, 1))


# 统计百分位数
def work2():
    print("统计百分位数")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(np.percentile(a, 50))
    print(np.percentile(a, 50, axis=0))
    print(np.percentile(a, 50, axis=1))


# 中位数、平均数
def work3():
    print("中位数、平均数")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 求中位数
    print(np.median(a))
    print(np.median(a, axis=0))
    print(np.median(a, axis=1))
    # 求平均数
    print(np.mean(a))
    print(np.mean(a, axis=0))
    print(np.mean(a, axis=1))


# 加权平均值
def work4():
    print("加权平均值")
    a = np.array([1, 2, 3, 4])
    weights = np.array([1, 2, 3, 4])
    print(np.average(a))
    print(np.average(a, weights=weights))


# 标准差、方差
def work5():
    print("标准差、方差")
    a = np.array([1, 2, 3, 4])
    print(np.std(a))
    print(np.var(a))


# 对数组进行排序
def work6():
    print("对数组进行排序")
    a = np.array([[4, 3, 2], [2, 4, 1]])
    print(np.sort(a))
    print(np.sort(a, axis=None))
    print(np.sort(a, axis=0))
    print(np.sort(a, axis=1))
    print(type(a))


# 对数组进行排序
def work7():
    print("对数组进行排序")
    # 使用List进行排序
    a = [4, 3, 2, 2, 4, 1]
    print(type(a))
    a.sort()
    print(a)
    a.sort(reverse=True)
    print(a)


work1()
work2()
work3()
work4()
work5()
work6()
work7()


# ndarray使用
def work8():
    print('ndarray使用')
    a = np.array([1, 2, 3])
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b[1, 1] = 10
    print(a.shape)
    print(b.shape)
    print(a.dtype)
    print(b)


# 结构化数组的使用
def work9():
    print('结构化数组的使用')
    persontype = np.dtype({'names': ['name', 'age', 'chinese', 'math', 'english'],
                           'formats': ['S32', 'i', 'i', 'i', 'f']})
    peoples = np.array([("ZhangFei", 32, 75, 100, 90), ("GuanYu", 24, 85, 96, 88.5), ("ZhaoYun", 28, 85, 92, 96.5),
                        ("HuangZhong", 29, 65, 85, 100)], dtype=persontype)
    ages = peoples['age']
    chineses = peoples['chinese']
    maths = peoples['math']
    englishs = peoples['english']
    print(np.mean(ages))
    print(np.mean(chineses))
    print(np.mean(maths))
    print(np.mean(englishs))


work8()
work9()


# 连续数组的创建：arange 或 linspace
def work10():
    print('连续数组的创建：arange 或 linspace')
    x1 = np.arange(1, 11, 2)
    x2 = np.linspace(1, 9, 5)
    print('x1=', x1)
    print('x2=', x2)

    print(np.add(x1, x2))
    print(np.subtract(x1, x2))
    print(np.multiply(x1, x2))
    print(np.divide(x1, x2))
    print(np.power(x1, x2))
    print(np.remainder(x1, x2))


work10()
