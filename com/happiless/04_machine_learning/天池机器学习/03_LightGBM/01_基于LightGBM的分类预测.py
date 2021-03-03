import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/high_diamond_ranked_10min.csv')
y = df['blueWins']

df.info()
df.head()
df.tail()
df.describe()
# 标注标签并利用value_counts函数查看训练集标签的数量
y.value_counts()

# 标注特征列
drop_cols = ['gameId', 'blueWins']
x = df.drop(drop_cols, axis=1)
x.describe()

# 根据上面的描述，我们可以去除一些重复变量，比如只要知道蓝队是否拿到一血，我们就知道红队有没有拿到，可以去除红队的相关冗余数据。
drop_cols = ['redFirstBlood', 'redKills', 'redDeaths',
             'redGoldDiff', 'redExperienceDiff', 'blueCSPerMin',
             'redCSPerMin', 'blueGoldPerMin', 'redGoldPerMin']
x.drop(drop_cols, axis=1, inplace=True)
x.head()

data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 0:9]], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# 绘制小提琴图
sns.violinplot(x='Features', y='Values', hue='blueWins', data=data, split=True, inner='quart', ax=ax[0],
               palette='Blues')
fig.autofmt_xdate(rotation=45)

data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 9:18]], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

sns.violinplot(x='Features', y='Values', hue='blueWins', data=data, split=True, inner='quart', ax=ax[1],
               palette='Blues')
fig.autofmt_xdate(rotation=45)

plt.figure(figsize=(12, 12))
sns.heatmap(round(x.corr(), 2), cmap='Blues', annot=True)

# 去除冗余特征
drop_cols = ['redAvgLevel', 'blueAvgLevel']
x.drop(drop_cols, axis=1, inplace=True)

sns.set(style='whitegrid', palette='muted')

# 构造两个新特征
x['wardsPlacedDiff'] = x['blueWardsPlaced'] - x['redWardsPlaced']
x['wardsDestroyedDiff'] = x['blueWardsDestroyed'] - x['redWardsDestroyed']
data = x[['blueWardsPlaced', 'blueWardsDestroyed', 'wardsPlacedDiff', 'wardsDestroyedDiff']].sample(1000)

data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

plt.figure(figsize=(10, 6))
sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)
plt.xticks(rotation=45)

# 去除和眼位相关的特征
drop_cols = ['blueWardsPlaced', 'blueWardsDestroyed', 'wardsPlacedDiff',
             'wardsDestroyedDiff', 'redWardsPlaced', 'redWardsDestroyed']
x.drop(drop_cols, axis=1, inplace=True)

x['killsDiff'] = x['blueKills'] - x['blueDeaths']
x['assistsDiff'] = x['blueAssists'] - x['redAssists']
x[['blueKills', 'blueDeaths', 'blueAssists', 'redAssists', 'assistsDiff', 'killsDiff']].hist(figsize=(12, 10), bins=20)

data = x[['blueKills', 'blueDeaths', 'blueAssists', 'redAssists', 'assistsDiff', 'killsDiff']].sample(1000)
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

# 我们发现击杀、死亡与助攻数的数据分布差别不大。但是击杀减去死亡、助攻减去死亡的分布与原分布差别很大，因此我们新构造这么两个特征
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)
plt.xticks(rotation=45)

# 从上图我们可以发现击杀数与死亡数与助攻数，以及我们构造的特征对数据都有较好的分类能力
data = pd.concat([y, x], axis=1)
sns.pairplot(data, vars=['blueKills', 'blueDeaths', 'blueAssists', 'killsDiff', 'assistsDiff', 'redAssists'],
             hue='blueWins')

# 一些特征两两组合后对于数据的划分能力也有提升。
x['dragonsDiff'] = x['blueDragons'] - x['redDragons']
x['heraldsDiff'] = x['blueHeralds'] - x['redHeralds']
x['eliteDiff'] = x['blueEliteMonsters'] - x['redEliteMonsters']
data = pd.concat([y, x], axis=1)
eliteGroup = data.groupby(['eliteDiff'])['blueWins'].mean()
dragonGroup = data.groupby(['dragonsDiff'])['blueWins'].mean()
heraldGroup = data.groupby(['heraldsDiff'])['blueWins'].mean()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
eliteGroup.plot(kind='bar', ax=ax[0])
dragonGroup.plot(kind='bar', ax=ax[1])
heraldGroup.plot(kind='bar', ax=ax[2])

x['towerDiff'] = x['blueTowersDestroyed'] - x['redTowersDestroyed']
data = pd.concat([y, x], axis=1)
towerGroup = data.groupby(['towerDiff'])['blueWins']

fig, ax = plt.subplots(1, 2, figsize=(15, 4))

towerGroup.mean().plot(kind='line', ax=ax[0])
ax[0].set_title('Proportion of Blue Wins')
ax[0].set_ylabel('Proportion')

towerGroup.count().plot(kind='line', ax=ax[1])
ax[1].set_title('Count of Towers Destroyed')
ax[1].set_ylabel('Count')

# 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split

# 选择其类别为0和1的样本 （不包括类别为2的样本）
data_target_part = y
data_features_part = x

# 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size=0.2,
                                                    random_state=2021)

## 导入LightGBM模型
from lightgbm.sklearn import LGBMClassifier

# 定义 LightGBM 模型
clf = LGBMClassifier()

# 在训练集上训练LightGBM模型
clf.fit(x_train, y_train)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

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

sns.barplot(y=data_features_part.columns, x=clf.feature_importances_)

from sklearn.metrics import accuracy_score
from lightgbm import plot_importance


def estimate(model, data):
    ax1 = plot_importance(model, importance_type='gain')
    ax1.set_title('gain')
    ax2 = plot_importance(model, importance_type='split')
    ax2.set_title('split')
    # plt.show()


def classes(data, label, test):
    model = LGBMClassifier()
    model.fit(data, label)
    ans = model.predict(test)
    estimate(model, data)
    return ans


ans = classes(x_train, y_train, x_test)
pre = accuracy_score(y_test, ans)
print('acc = ', pre)

# 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV

# 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
feature_fraction = [0.5, 0.8, 1]
num_leaves = [16, 32, 64]
max_depth = [-1, 3, 5, 8]

parameters = {'learning_rate': learning_rate,
              'feature_fraction': feature_fraction,
              'num_leaves': num_leaves,
              'max_depth': max_depth}

model = LGBMClassifier(n_estimators=50)

# 进行网格搜索
cv_clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy_score', verbose=3, n_jobs=-1)
cv_clf = cv_clf.fit(x_train, y_train)

# 网格搜索后的最好参数为
print(cv_clf.best_params_)
