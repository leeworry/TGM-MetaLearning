# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 8}
plt.rc('font', **font)
#X轴，Y轴数据
y1 = [28.6,28.7,28.1,27.7,27.7]
y2 = [20.2,20.3,19.5,18.9,19]
y3 = [30.8,30.9,30.7,30.7,30.4]
fig = plt.figure(figsize=(4,3)) #创建绘图对象,dpi=400
ax1 = fig.add_subplot(1,1,1)
x = [1,2,3,4,5]
x_index = list(range(1,len(x)+1))
ax1.set_xticks([10,20,50,100,200])
ax1.set_yticks([19,21,23,25,27,29,31])

MRR, = plt.plot(x_index,y1,"black",linewidth=1.5,linestyle='-',marker="^",color="#2298D9",markersize=3,markerfacecolor='white')
H1, = plt.plot(x_index,y2,"black",linewidth=1.5,linestyle='-',marker="s",color="#E6B33D",markersize=3,markerfacecolor='white')
H3, = plt.plot(x_index,y3,"black",linewidth=1.5,linestyle='-',marker=".",color="#407434",markersize=7,markerfacecolor='white')

plt.legend([MRR, H1, H3], ["MRR","H@1","H@3"],loc=0,frameon=True,prop = {'size':6},ncol=1)
plt.xticks(x_index,x)
# plt.xlabel("$K$") #X轴标签
# plt.ylabel("(%)") #Y轴标签
plt.grid(axis='y',linestyle='--')
plt.grid(axis='x',linestyle='--')
plt.show() #显示图
plt.savefig("res15acc.jpg") #保存图