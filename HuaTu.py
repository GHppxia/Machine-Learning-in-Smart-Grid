import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'simhei'

#读取目标表格文件，并用people代表读取到的表格数据
people = pd.read_csv('avg.csv')
#x轴是姓名，y轴是年龄，让直方图排序显示，默认升序
people.sort_values(y='precision',inplace=True,ascending=False)
#在控制台中输出表格数据
print(people)
#将直方图颜色统一设置为蓝色
people.plot.bar(x='classifier',y='precision',color='brown')
#旋转X轴标签，让其横向写 蓝色：blue,橙色：orange
#for a,b in zip(x,y)
plt.xticks(rotation=360)
plt.show()
