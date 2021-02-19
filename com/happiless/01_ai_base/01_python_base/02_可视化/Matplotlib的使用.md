## Matplotlib的使用

>Matplotlib 是一个 [Python](https://baike.baidu.com/item/Python) 的 2D绘图库。通过 Matplotlib，开发者可以仅需要几行代码，便可以生成绘图，直方图，功率谱，条形图，错误图，散点图等。

##### 为什么学习Matplotlib

- 可让数据可视化，更直观的真实给用户。使数据更加客观、更具有说服力。
- Matplotlib是Python的库，又是开发中常用的库

### Matplotlib的安装

`pip install matplotlib`

### Matplotlib的基本使用

```Python
import numpy as np 
from matplotlib import pyplot as plt 
 
x = np.arange(1,11) 
y =  2  * x +  5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y)
plt.show()
```

以上代码中，我们方法说明:

- plt.title() 设置图表的名称

- plt.xlabel() 设置x轴名称

- plt.ylabel() 设置y轴名称

- plt.xticks(x,ticks,rotation) 设置x轴的刻度,rotation旋转角度

- plt.yticks() 设置y轴的刻度                      

- plt.plot()  绘制线性图表

- plt.show() 显示图表

- plt.legend() 显示图例

- plt.text(x,y,text) 显示每条数据的值  x,y值的位置

- plt.figure(name,figsize=(w,h),dpi=n) 设置图片大小

  

### 图表中文显示

Matplotlib 默认情况不支持中文，我们可以使用以下简单的方法来解决：

首先下载字体（注意系统）：<https://www.fontpalace.com/font-details/SimHei/>

#### 方法1：引入字体文件

```python
zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf")
plt.title("图表 - 测试",fontproperties=zhfont1)
```

#### 方法2：使用系统文字

```python
# 查看系统支持的字体
a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
for i in a:
    print(i)
```

在上面的代码中找到可以使用的字体，使用

```python
plt.rcParams['font.family']=['SimHei']
```



### 线性图表

​	以折线的上升或下降来表示统计数量的增减变化的统计图

​	特点：能够显示数据的变化走势，反映事物的**变化**情况

- plt.plot(x,y,type,label)

  - x 显示的数据

  - y y轴的值

  - type 值显示的方式  具体值如下

  - label 图例名称

    | 字符       | 描述         |
    | ---------- | ------------ |
    | `'-'`      | 实线样式     |
    | `'--'`     | 短横线样式   |
    | `'-.'`     | 点划线样式   |
    | `':'`      | 虚线样式     |
    | `'.'`      | 点标记       |
    | `','`      | 像素标记     |
    | `'o'`      | 圆标记       |
    | `'v'`      | 倒三角标记   |
    | `'^'`      | 正三角标记   |
    | `'&lt;'`   | 左三角标记   |
    | `'&gt;'`   | 右三角标记   |
    | `'1'`      | 下箭头标记   |
    | `'2'`      | 上箭头标记   |
    | `'3'`      | 左箭头标记   |
    | `'4'`      | 右箭头标记   |
    | `'s'`      | 正方形标记   |
    | `'p'`      | 五边形标记   |
    | `'*'`      | 星形标记     |
    | `'h'`      | 六边形标记 1 |
    | `'H'`      | 六边形标记 2 |
    | `'+'`      | 加号标记     |
    | `'x'`      | X 标记       |
    | `'D'`      | 菱形标记     |
    | `'d'`      | 窄菱形标记   |
    | `'&#124;'` | 竖直线标记   |
    | `'_'`      | 水平线标记   |

    颜色如下：

    | 字符  | 颜色   |
    | ----- | ------ |
    | `'b'` | 蓝色   |
    | `'g'` | 绿色   |
    | `'r'` | 红色   |
    | `'c'` | 青色   |
    | `'m'` | 品红色 |
    | `'y'` | 黄色   |
    | `'k'` | 黑色   |
    | `'w'` | 白色   |

    ```pyhon
    plt.plot(x, y,'*m')
    ```





### 绘画条状图

排列在工作表的列或行中的数据可以绘制到中

特点：绘制连**离散**的数据，能够一眼看出各个数据的大小，可以快速**统计**数据之间的差别

 bar(x,y,color,width) 函数来生成纵向条形图

 barh(x,y,color,height) 函数来生成条形图

- x 条装显示位置
- y 显示的值
- color 显示的颜色

```python
from matplotlib import pyplot as plt 
x =  [5,8,10] 
y =  [12,16,6] 
x2 =  [6,9,11] 
y2 =  [6,15,7] 
plt.bar(x, y, align =  'center') 
plt.bar(x2, y2, color =  'g', align =  'center') 
plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis') 
plt.show()
```



### 绘画直方图

由一系列高度不等的纵向条纹或线段表示数据分布的情况，一般用横轴表示数据范围，纵轴表示分布情况

特点：绘制连续性的数据，展示一组或多组数据的分布状况并**统计**

注意：拿到数据来统计，而不是直接拿统计好的数据

概念：

​	组距：每组数据的分割区域，例如1-5一组5-10一组。我们可以称数据的组距为5

​	组数：(最大数据-最小数据)/组距 一般会100条数据可分5-12组

hist(data,bins,normed)

- data 所有的数据
- bins 分几组
- normed y轴是否显示成百分比

```python
plt.hist(data,bins)
```

### 绘画散点图

用两组数据构成多个坐标点，考察坐标点的分布，判断两变量之间是否存在某种关联或总结坐标点的分布模式

```python
plt.scatter(x,info)
plt.plot(x,a,'o')
```

特点：判断变量之间是否存在在数量关联走势，展示离群点**分布规律**

### 绘画子图

subplot() 函数允许你在同一图中绘制不同的东西

``` python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(7, 7))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])

plt.show()
```