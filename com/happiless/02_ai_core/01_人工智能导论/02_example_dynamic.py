from collections import defaultdict
from functools import wraps

original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]
price = defaultdict(int)
for i, p in enumerate(original_price):
    price[i+1] = p


def memo(func):
    _cache = {}
    @wraps(func)
    def _wrap(n):
        if n in _cache: result = _cache[n]
        else:
            result = func(n)
            _cache[n] = result
        return result
    return _wrap


@memo
def r(n):
    max_price, split_point = max([(price[n], 0)] + [(r(i) + r(n-i), i) for i in range(1, n)], key=lambda x:x[0])
    solution[n] = (split_point, n - split_point)
    return max_price


def not_cut(split): return split == 0


def parse_solution(target_length, revenue_solution):
    left, right = revenue_solution[target_length]

    if not_cut(left): return [right]

    return parse_solution(left, revenue_solution) + parse_solution(right, revenue_solution)


if __name__ == '__main__':
    solution = {}
    print(r(5))
    print(price)
    print(solution)
    print(parse_solution(5, solution))
