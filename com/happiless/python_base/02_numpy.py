# numpy
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print("数组a的维度个数", a.ndim)
print("数组a的各个维度长度", a.shape)
print("数组a的元素总数", a.size)
print("数组a的元素类型", a.dtype)

b = np.arange(0, 20, 5)
print(b.shape)
print(b.reshape(2, 2))
c = np.linspace(0, 2, 10)
print(c)

zero_arr = np.zeros((3, 4))
one_arr = np.ones((2, 3, 4), dtype=np.int64)
print(zero_arr)
print(one_arr)

print(np.eye(3))  # 单位矩阵

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(arr1)
print(arr2)
print(arr2 - arr1)
print(arr1 + arr2)
print(arr1 * arr2)
print(arr2 / arr1)
# 1 2      5 6
# 3 4      7 8
# 1*5+2*7 1*6+2*8
# 3*5+4*7 3*6+4*8
# 19 22
# 43 50
print(np.dot(arr1, arr2))
print(arr1.dot(arr2))
print(arr1.T)

arr3 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr3)
print("arr3的最大元素索引", np.argmax(arr3))
print("arr3沿第一个轴的最大元素索引", np.argmax(arr3, axis=0))
print("arr3沿第二个轴的最大元素索引", np.argmax(arr3, axis=1))

# 求解线性方程组
a_arr = np.array([1, 2])
print("array:", a_arr)
print("array shape:", a_arr.shape)
a_mat = np.mat([1, 2])
print("mat:", a_mat)
print("mat shape:", a_mat.shape)

