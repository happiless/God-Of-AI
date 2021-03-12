#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 01_test_logistic.py    
# 库函数导入
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Step2:模型训练
# 构造数据集
x_features = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])
# 调用逻辑回归模型
lr_clf = LogisticRegression()
# 用逻辑回归模型拟合构造的数据集
lr_clf.fit(x_features, y_label)
# 查看其对应模型的w
print('the weight of Logistic Regression:', lr_clf.coef_)
# 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:', lr_clf.intercept_)

# Step4:数据和模型可视化
# 可视化构造的数据样本点
plt.figure()
plt.scatter(x_features[:, 0], x_features[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
plt.show()

# 可视化决策边界
plt.figure()
plt.scatter(x_features[:, 0], x_features[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
z_proba = z_proba[:, 1].reshape(x_grid.shape)
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

# 可视化预测新样本
# new point 1
plt.figure()
x_features_new1 = np.array([[0, -1]])
plt.scatter(x_features_new1[:, 0], x_features_new1[:, 1], s=50, cmap='viridis')
plt.annotate('New point 1', xy=(0, -1), xytext=(-2, 0), color='blue',
             arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='red'))

# new point 2
x_features_new2 = np.array([[1, 2]])
plt.scatter(x_features_new2[:, 0], x_features_new2[:, 1], s=50, cmap='viridis')
plt.annotate('New point 2', xy=(1, 2), xytext=(-1.5, 2.5), color='red',
             arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='red'))

# 训练样本
plt.scatter(x_features[:, 0], x_features[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')

# 可视化决策边界
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

plt.show()

# Step5:模型预测
# 在训练集和测试集上分别利用训练好的模型进行预测
y_label_new1_predict = lr_clf.predict(x_features_new1)
y_label_new2_predict = lr_clf.predict(x_features_new2)
print('The New point 1 predict class:\n', y_label_new1_predict)
print('The New point 2 predict class:\n', y_label_new2_predict)

# 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所以我们可以利用 predict_proba 函数预测其概率
y_label_new1_predict_proba = lr_clf.predict_proba(x_features_new1)
y_label_new2_predict_proba = lr_clf.predict_proba(x_features_new2)
print('The New point 1 predict Probability of each class:\n', y_label_new1_predict_proba)
print('The New point 2 predict Probability of each class:\n', y_label_new2_predict_proba)
