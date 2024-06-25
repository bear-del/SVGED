import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

size = 4
a = [98.507, 97.916, 98.299, 90.903, 87.500] #OCSVM
b = [99.861, 98.194, 98.437, 90.903, 87.500] #SVDD
c = [99.373, 99.081, 63.420, 42.077, 14.512] #IF
d = [98.750, 99.248, 82.361, 37.535, 37.188] #LOF
# e = [86.52, 100, 98.58, 60.61, 90.68, 82.76] #l=3 d=9
# f = [84.67, 100, 96.78, 62.41, 89.41, 74.76] #l=3 d=12

Trans = np.vstack((a,b,c,d)).T
typeA = Trans[0]
typeB = Trans[1]
typeC = Trans[2]
typeD = Trans[3]
typeE = Trans[4]
# typeF = Trans[5]
x = np.arange(size)

# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.6, 5
# width1 = 0.2
# width2 = 0.15
# 每种类型的柱状图宽度
width = total_width / n

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
print(x)

# 画柱状图
plt.bar(x, typeA, width=width, label="1",edgecolor='k',color='darkorange')
plt.bar(x + width, typeB, width=width, label="2",edgecolor='k',color='skyblue')
plt.bar(x + 2*width, typeC, width=width, label="3",edgecolor='k',color='lightblue')
plt.bar(x + 3*width, typeD, width=width, label="4",edgecolor='k',color='paleturquoise')
plt.bar(x + 4*width, typeE, width=width, label="5",edgecolor='k',color='lightcyan')
# plt.bar(x + 5*width, typeF, width=width, label="f",edgecolor='k',color='lightcyan')

#
name_list = ('OCSVM', 'SVDD', 'IF', 'LOF')
value_list = np.random.randint(0, 99, size = len(name_list))
pos_list = np.arange(len(name_list))
plt.xticks(pos_list, name_list)

# 显示图例
# plt.legend()
# 显示柱状图
plt.savefig('./final/graph/bar_2.png', dpi=500)
plt.show()
