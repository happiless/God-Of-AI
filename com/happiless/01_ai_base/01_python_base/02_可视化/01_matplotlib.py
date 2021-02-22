import matplotlib.pyplot as plt


# matplotlib标签
def test01():
    max_temperature = [26, 30, 31, 32, 33]
    min_temperature = [12, 16, 16, 17, 18]

    x = range(5)

    x_ticks = [f"星期{i}" for i in range(1, 6)]
    plt.rcParams['font.family'] = ['SimHei']
    plt.title('某年某周第N周的温度')
    plt.xlabel('周')
    plt.ylabel('温度：单位(℃)')
    plt.tick_params(top=True, right=True, left=True, bottom=True)
    plt.grid(alpha=0.2)  # 透明度0-1之间
    plt.xticks(x, x_ticks)
    y = range(12, 35, 2)
    plt.yticks(y)

    plt.plot(x, max_temperature, label='最高温')
    plt.plot(x, min_temperature, label='最低温')

    plt.legend(loc=2)

    plt.show()

if __name__ == '__main__':
    test01()