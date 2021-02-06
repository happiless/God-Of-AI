#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
        yield a


for i in fibonacci(10):
    print(i)

print(repr(fibonacci(10)))

print(ord('A'))
print(chr(65))
print(chr(ord('A') + 32))

print(eval("2**3"))

import os
import platform

print(platform.architecture())
print(platform.uname())

import random

print(random.choices([1, 2, 3, 4]))


# 微信发红包
def repack(total, nums):
    if type(total) is not int or type(nums) is not int:
        return "Wrong numbers!"
    every = [0]
    for i in range(nums - 1):
        money = random.randint(1, total - sum(every)) if total - sum(every) > 0 else 0
        every.append(money)
    return every[1:] + [total - sum(every)]


print(repack(100, 4))

# python 小数精准运算
from decimal import Decimal

a = Decimal('4.2')
b = Decimal('2.1')
a + b
Decimal('6.3')
print(a + b)
print((a + b) == Decimal('6.3'))
