#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 02_iris_classify_by_logistic.py

# Step1:库函数导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step2:数据读取/载入
from sklearn.datasets import load_iris

data = load_iris()
iris_target = data['target']
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)

# Step3:数据信息简单查看
iris_features.info()
iris_features.head()  # 头5行
iris_features.tail()  # 后5行
# 对于特征进行一些统计描述, 从统计描述中我们可以看到不同数值特征的变化范围。
iris_features.describe()
pd.Series(iris_target).value_counts()

# Step4:可视化描述
# 合并标签和特征信息
iris_all = iris_features.copy()  # 进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target

# 特征与标签组合的散点可视化
sns.pairplot(data=iris_all, diag_kind='hist', hue='target')
plt.show()

for col in iris_features.columns:
    sns.boxplot(x='target', y=col, saturation=0.5, palette='pastel', data=iris_all)
    plt.title(col)
    plt.show()

# 选取其前三个特征绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
iris_all_class0 = iris_all[iris_all['target'] == 0].values
iris_all_class1 = iris_all[iris_all['target'] == 1].values
iris_all_class2 = iris_all[iris_all['target'] == 2].values

ax.scatter(iris_all_class0[:, 0], iris_all_class0[:, 1], iris_all_class0[:, 2], label='setosa')
ax.scatter(iris_all_class1[:, 0], iris_all_class1[:, 1], iris_all_class1[:, 2], label='versicolor')
ax.scatter(iris_all_class2[:, 0], iris_all_class2[:, 1], iris_all_class2[:, 2], label='virginica')
plt.legend()

plt.show()

# Step5:利用 逻辑回归模型 在二分类上 进行训练和预测
# 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split

# 选择其类别为0和1的样本 （不包括类别为2的样本）
iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]

# 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size=0.2,
                                                    random_state=2020)

# 从sklearn中导入逻辑回归模型
from sklearn.linear_model import LogisticRegression

# 定义 逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')

# 在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)

# 查看其对应的w
print('the weight of Logistic Regression:', clf.coef_)
# 查看其对应的w0
print('the intercept(w0) of Logistic Regression:', clf.intercept_)

# 预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 评估
from sklearn import metrics

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))

# 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Step6:利用 逻辑回归模型 在三分类(多分类)上 进行训练和预测
x_train, x_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size=0.2, random_state=2020)
clf = LogisticRegression(random_state=0, solver='lbfgs')
clf.fit(x_train, y_train)

print('the weight of Logistic Regression:\n', clf.coef_)
print('the intercept(w0) of Logistic Regression:\n', clf.intercept_)

train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用 predict_proba 函数预测其概率
train_predict_proba = clf.predict_proba(x_train)
test_predict_proba = clf.predict_proba(x_test)
print('The test predict Probability of each class:\n', test_predict_proba)
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))

# 查看混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
