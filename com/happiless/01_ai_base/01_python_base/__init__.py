# 万年历
import calendar, time, os


# 每个月份的日历信息的基础上，可以完成上一月和下一月信息的显示，
# 并且可以指定年或月

def showdate(year, month):
    res = calendar.monthrange(year, month)
    days = res[1]  # 当前月份的天数
    w = res[0] + 1  # 当前月份第一天周几信息
    print(f'====={year}年{month}月的日历信息=====')
    print(' 一   二  三  四   五  六  日 ')
    print('*' * 28)
    # 实现日历信息的输出
    d = 1
    while d <= days:
        # 循环周
        for i in range(1, 8):
            # 判断是否输出
            if d > days or (i < w and d == 1):
                print(' ' * 4, end="")
            else:
                print(' {:0>2d} '.format(d), end="")
                d += 1
        print()
    print('*' * 28)


# 获取当前系统的年，月
dd = time.localtime()
year = dd.tm_year  # 获取年
month = dd.tm_mon  # 获取月

while True:
    # os.system('clear')
    # 默认输出当前年月的日历信息
    showdate(year, month)
    print('< 上一月   下一月 >')
    # 获取用户的输入
    c = input('请输入您的选择：')
    # 判断用户的输入内容
    if c == '<':
        month -= 1
        if month < 1:
            month = 12
            year -= 1
    elif c == '>':
        month += 1
        if month > 12:
            month = 1
            year += 1
    else:
        print('输入内容错误，请重新输入')
