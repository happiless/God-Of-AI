#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 01_homework_dynamic_example.py
# @todo    : 动态规划问题
from collections import defaultdict
from functools import wraps


def memo(func):
    _cache = {}

    @wraps(func)
    def _warp(n):
        if n in _cache:
            result = _cache[n]
        else:
            result = func(n)
            _cache[n] = result
        return result

    return _warp


@memo
def r(n):

    # candidates = []
    # for i in range(1, n):
    #     candidates.append((r(i) + r(n - i), i))
    # candidates.append((price[n], 0))
    # max_price, split_point = max(candidates, key=lambda x: x[0])

    max_price, split_point = max([(price[n], 0)] + [(r(i) + r(n-i), i) for i in range(1, n)], key=lambda x: x[0])

    solution[n] = [split_point, n - split_point]
    return max_price


def not_cut(n): return n == 0


def parse_solution(target, revenue_solution):
    left, right = revenue_solution[target]

    if not_cut(left): return [right]

    return parse_solution(left, revenue_solution) + parse_solution(right, revenue_solution)


if __name__ == '__main__':
    original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]

    price = defaultdict(int)
    for i, p in enumerate(original_price):
        price[i + 1] = p
    solution = {}
    r(100)
    print(solution)
    print(parse_solution(8, solution))
