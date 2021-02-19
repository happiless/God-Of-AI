## Seaborn的使用

### 简介

`Seaborn` 是以 `matplotlib`为底层，更容易定制化作图的`Python`库。

`Seaborn` 其实是在`matplotlib`的基础上进行了更高级的API封装，从而使得作图更加容易。

在大多数情况下使用`Seaborn`就能做出很具有吸引力的图，而使用matplotlib就能制作具有更多特色的图，换句话说，``matplotlib``更加灵活，可定制化，而`seaborn`像是更高级的封装，使用方便快捷。

应该把`Seaborn`视为`matplotlib`的补充，而不是替代物。



### 安装

```shell
pip install seaborn
```



### 背景风格管理

除了各种绘图方式外，图形的美观程度可能是我们最关心的了。将它放到第一部分，因为风格设置是一些通用性的操作，对于各种绘图方法都适用。

`Seaborn`支持的风格有5种：

- **darkgrid**  黑背景-白格
- **whitegrid** 白背景-白格
- **dark**  黑背景
- **white** 白背景
- **ticks** 

设置风格的方法：

- set()
- set_style(value) 统一设置
- axes_style(value) 单一设置

```python
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# 获取数据
xiao_mi = np.random.randint(50, 100, size=10)
hua_wei = np.random.randint(50, 100, size=10)
x = range(10)

with sns.axes_style('darkgrid'):
    plt.subplot(2,3,1)
    plt.plot(x, xiao_mi)
with sns.axes_style('whitegrid'):
    plt.subplot(2,3,2)
    plt.plot(x, xiao_mi)
with sns.axes_style('dark'):
    plt.subplot(2,3,3)
    plt.plot(x, xiao_mi)
with sns.axes_style('white'):
    plt.subplot(2,3,4)
    plt.plot(x, xiao_mi)
    
sns.set_style('ticks')
plt.subplot(2,3,5)
plt.plot(x, xiao_mi)
plt.show()
```

**注意事项：**  修改样式要在填充数据之前

### 移除轴脊柱

**white** 和 **ticks**两个风格都能够移除顶部和右侧的不必要的轴脊柱。通过`matplotlib`参数是做不到这一点的，但是你可以使用`seaborn`的`despine()`方法来移除它们

```
sns.despine()
```

**注意事项：** 移除轴脊柱要在 填充数据之后，在显示图表之前

### 图像风格管理

这会影响标签的大小，线条和绘图的其他元素，但不会影响整体样式。一组独立的参数控制绘图元素的比例，这应该允许您使用相同的代码来制作适合在适合使用较大或较小绘图的设置中的绘图。

首先让我们通过调用重置默认参数[`set()`](http://seaborn.pydata.org/generated/seaborn.set.html#seaborn.set)：

- set()
- set_context()

四个预设上下文中，在相对大小的顺序，是`paper`，`notebook`，`talk`，和`poster`。该`notebook`样式是默认的，并且在上面的图表中使用。

```
sns.set_context("paper")

sinplot()
```

### 调色板

颜色比图形样式的其他方面更重要，因为如果有效使用颜色可以更凸显示数据的结果与重要

Seaborn可以轻松选择和使用适合您正在使用的数据类型的调色板以及您可视化的目标

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
```

- color_palette()能传入任何Matplotlib所支持的颜色
- color_palette()不写参数则,默认颜色 deep, muted, pastel, bright, dark, colorblind(圆形颜色系统)
- color_palette() flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"] 
- light_palette()
- dark_palette()
- set_palette()设置所有图的颜色

#### 调色板

```python
current_palette = sns.color_palette()
sns.palplot(current_palette)
plt.show()
```



![](https://note.youdao.com/yws/api/personal/file/WEB6e199eb6707cf78ead5112b9d0108c68?method=download&shareKey=e153fe7e08c25e40bfb76f305fbdb0f5)



```python
current_palette = sns.color_palette()
sns.palplot(current_palette) #12个色块
sns.palplot(sns.color_palette('hls',8))#8个颜色，色度比较亮
plt.show()
```



![](https://note.youdao.com/yws/api/personal/file/WEBc69497b873583d3301d3f8ec9c8a8439?method=download&shareKey=399897c141ecc19d2dff495c92bee6c3)



![](https://note.youdao.com/yws/api/personal/file/WEB680e0e3b2c4ecf55aaaec136a82eed8d?method=download&shareKey=e440b7cb18023d523de637b14e84ad7b)

#### 连续的调色板

```python
sns.palplot(sns.color_palette('Blues'))
sns.palplot(sns.color_palette('Greens_r')) #参数里可以输入数字，指定颜色的块数,_r由深到浅
plt.show()
```



![](https://note.youdao.com/yws/api/personal/file/WEB52aa4b9c53bb690a2fc383786188a295?method=download&shareKey=9d802e5dda753e178c706bc91ec788d7)

![](https://note.youdao.com/yws/api/personal/file/WEB15f6e557a3ee10492502cac64c6b684e?method=download&shareKey=9aa4e6600f3d78722562d15d7ea74aca)

#### 自定义连续调色板

```python
sns.palplot(sns.light_palette('green'))
sns.palplot(sns.dark_palette('purple'))
plt.show()
```

![](https://note.youdao.com/yws/api/personal/file/WEBf73413bda4c9e698bbef0825b4bfd5ae?method=download&shareKey=e496889a241a4fada5d13582ffd5cf59)

![](https://note.youdao.com/yws/api/personal/file/WEB53e3707b620f32b1548451be717db122?method=download&shareKey=967f1926c2cf951fb768267ec75e5d57)



#### xkcd_rgb 颜色

Xkcd_rgb这产生了一组[954种命名颜色](https://xkcd.com/color/rgb/)，您现在可以使用`xkcd_rgb`在seaborn中引用它们

```python
plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)
plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
plt.plot([0, 1], [0, 3], sns.xkcd_rgb["denim blue"], lw=3)
```

#### 线性调色板

#### 色板的应用

```python
sns.set_style('whitegrid') #dark;white;
data = np.random.normal(size=(20,8))+np.arange(8)/2 #正态分布 loc均值，scale标准差，size20行8列数据
sns.boxplot(data=data,palette=sns.color_palette('hls',8))
plt.show()
```

```python
plt.bar(x,x,color=sns.color_palette('Reds'))
plt.show()
```

### 单变量

当我们做分析时，一般会看下单数据的分布情况，也就是但变量的分析与统计。

#### distplot 直方图

```python
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
x = np.random.normal(size=100)
sns.distplot(x,kde=False)
#sns.distplot(x, bins=20, kde=False) 设置多少个分组
sns.distplot(x, kde=False)
```

#### jointplot 散点图

```python
# 生成 多元高斯正太分布 的数据
# 多元高斯正太分布是一堆正太分布向更多维度的推广，这种分布由其均值和协方差矩阵来确定
data = np.random.multivariate_normal([0,1], [(1,0.5),(0.5,1)], 1000) # 均值， [(1,0.5),(0.5,1)]协方差对称阵
df = pd.DataFrame(data,columns=['x','y'])
with sns.axes_style("white"):
    sns.jointplot(x='x', y='y',data=df)
plt.show()
```

#### pairplot

该函数会同时绘制数据中所有特征两两之间的关系图.因为pairplot是建立在pairgrid之上,所以可以将中间的很多函数进行变换,例如下面的kde的例子.

 默认对角线histgram，非对角线kdeplot

```python
iris=sns.load_dataset('iris')  #导入经典的鸢尾花数据
sns.pairplot(iris);
```

### 回归分析图

当得到数据时，我可以对连续性的值，可以做一些回归关系的绘制与展示，具体做法如下

- sns.lmplot() 功能多，规范多
- sns.regplot() 推荐，支持的参数与数据类型比较多一些

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# 准备数据
tips = sns.load_dataset("tips")
tips.head()
```

```python
sns.regplot(x="total_bill", y="tip", data=tips)

sns.lmplot(x="total_bill", y="tip", data=tips)

sns.regplot(data=tips,x="size",y="tip")

sns.regplot(x="size", y="tip", data=tips, x_jitter=.05)
```



### 多变量分析绘图

```python
# 准备数据
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")
```

```python
sns.stripplot(x="day", y="total_bill", data=tips)

sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
# 花装图
sns.swarmplot(x="day", y="total_bill", data=tips)

sns.swarmplot(x="day", y="total_bill", hue="sex",data=tips)

sns.swarmplot(x="total_bill", y="day", hue="time", data=tips);
```



```python
# 盒图
sns.boxplot(x="day", y="total_bill", hue="time", data=tips)
# 小提琴图
sns.violinplot(x="total_bill", y="day", hue="time", data=tips)

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)
# 结合使用
sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5)


```



```python
titanic= sns.load_dataset('titanic')
#条形图
sns.barplot(x="sex", y="survived", hue="class", data=titanic)
# 点图
sns.pointplot(x="sex", y="survived", hue="class", data=titanic)

sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g", "female": "m"},
              markers=["^", "o"], linestyles=["-", "--"])

```



```python
# 多功能方法
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")

sns.factorplot(x="day", y="total_bill", hue="smoker",
               col="time", data=tips, kind="swarm")

sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=.5)


```

> seaborn.factorplot(x=None, y=None, hue=None, data=None, row=None, col=None, col_wrap=None, estimator=, ci=95, n_boot=1000, units=None, order=None, hue_order=None, row_order=None, col_order=None, kind='point', size=4, aspect=1, orient=None, color=None, palette=None, legend=True, legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, **kwargs)

Parameters：

- x,y,hue 数据集变量 变量名
- date 数据集 数据集名
- row,col 更多分类变量进行平铺显示 变量名
- col_wrap 每行的最高平铺数 整数
- estimator 在每个分类中进行矢量到标量的映射 矢量
- ci 置信区间 浮点数或None
- n_boot 计算置信区间时使用的引导迭代次数 整数
- units 采样单元的标识符，用于执行多级引导和重复测量设计 数据变量或向量数据
- order, hue_order 对应排序列表 字符串列表
- row_order, col_order 对应排序列表 字符串列表
- kind : 可选：point 默认, bar 柱形图, count 频次, box 箱体, violin 提琴, strip 散点，swarm 分散点 size 每个面的高度（英寸） 标量 aspect 纵横比 标量 orient 方向 "v"/"h" color 颜色 matplotlib颜色 palette 调色板 seaborn颜色色板或字典 legend hue的信息面板 True/False legend_out 是否扩展图形，并将信息框绘制在中心右边 True/False share{x,y} 共享轴线 True/False

###  FacetGrid

当想把数据中的多个子集展示出来时，可以使用FacetGrid，步骤如下：

- 绘画区域
- 使用方法绘画数据

FacetGrid常用参数：

- height  高
- aspect  宽
- palette 色板
- col 分图属性(列)
- row 分图属性(行)
- hue 属性分类
- margin_titles 
- size 图的大小，已被height 替代
- row_order 显示分图的数据
- hue_kws 显示图标记的形状(散点)

准备数据：

```python
tips = sns.load_dataset("tips")
tips.head()
```

绘图

```python
# 绘画区域
g = sns.FacetGrid(tips, col="time")
#----------------- 绘画直方图-----------------
g.map(plt.hist, "tip");

#----------------- 绘画散点图-------------------
g = sns.FacetGrid(tips, col="sex", hue="smoker") # 根据性别分类多图， 显示出2部分数据
g.map(plt.scatter, "total_bill", "tip", alpha=.7) 
g.add_legend(); #增加图例

#----------------- 绘画散点图-------------------
pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend();

#----------------- 绘画散点图-------------------
g = sns.FacetGrid(tips, hue="sex", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")
g.add_legend();


#------------------回归分析图-----------------
g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".1", fit_reg=False, x_jitter=.1)

#------------------条形图-----------------
g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
g.map(sns.barplot, "sex", "total_bill");

#------------------盒形图-----------------
from pandas import Categorical
ordered_days = tips.day.value_counts().index
print (ordered_days)
ordered_days = Categorical(['Thur', 'Fri', 'Sat', 'Sun'])
g = sns.FacetGrid(tips, row="day", row_order=ordered_days,
                  size=1.7, aspect=4,)
g.map(sns.boxplot, "total_bill");




with sns.axes_style("white"):
    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5);
g.set_axis_labels("Total bill (US Dollars)", "Tip");
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
g.fig.subplots_adjust(wspace=.02, hspace=.02);
#g.fig.subplots_adjust(left  = 0.125,right = 0.5,bottom = 0.1,top = 0.9, wspace=.02, hspace=.02)



```



### 热力图

热力图可以显示多个数据的走势与规律，并可以快速到到大值的与最小值所在位置

```python
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# 注备数据
uniform_data = np.random.rand(3, 3)
print (uniform_data)
heatmap = sns.heatmap(uniform_data)

ax = sns.heatmap(uniform_data, vmin=0.2, vmax=0.5) #设置最大值，也最小值

normal_data = np.random.randn(3, 3)
print (normal_data)
ax = sns.heatmap(normal_data, center=0)


# 航班数据
flights = sns.load_dataset("flights")
flights.head()
# 转换数据
flights = flights.pivot("month", "year", "passengers")
print (flights)
ax = sns.heatmap(flights)
# 绘画数据
ax = sns.heatmap(flights, annot=True,fmt="d") # 显示数字，并以10进制显示

ax = sns.heatmap(flights, linewidths=.5) # 设置边距

ax = sns.heatmap(flights, cmap="YlGnBu") # 设置颜色
ax = sns.heatmap(flights, cbar=False) #设置图例
```

