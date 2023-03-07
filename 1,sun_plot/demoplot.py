import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

## 分辨率 1500*1200
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

df = pd.read_csv("real.csv")
df_PSO = pd.read_csv("PSO_pred.csv")
df_TSA = pd.read_csv('TSA_pred.csv')
df_regular = pd.read_csv('regular_pred.csv')

fig, ax = plt.subplots(figsize =(8*np.sqrt(2)+6, 8)) # 创建图实例

ax.plot(np.linspace(0,df.shape[0]+1,df.shape[0]),df.iloc[:, -1],label='real value',marker='+')
ax.plot(np.linspace(0,df_regular.shape[0]+1,df_regular.shape[0]),df_regular.iloc[:,-1],label='LSTM predicted value',marker='2')
ax.plot(np.linspace(0,df_PSO.shape[0]+1,df_PSO.shape[0]),df_PSO.iloc[:,-1],label='PSO_LSTM predicted value',marker='1')
ax.plot(np.linspace(0,df_TSA.shape[0]+1,df_TSA.shape[0]),df_TSA.iloc[:,-1],label='TSA_LSTM predicted value',marker='*')


ax.set_xlabel('') #设置x轴名称 x label
ax.set_ylabel('归一化之后的目标特征') #设置y轴名称 y label
ax.set_title('太阳功率') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示

plt.show() #图形可视化
