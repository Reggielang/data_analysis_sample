# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:29:47 2020

@author: REGGIE
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
#修改字体显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


#%%
os.chdir(r'C:\Users\REGGIE\Desktop\数据分析资料\280_Ben_八大直播八大案例配套课件\实例')
df = pd.read_csv('teleco_camp_orig.csv')
df.head()

#%%-通过画图来找出离群值，或者错误值(培养数据的感觉)
plt.hist(df['AvgIncome'],bins = 20, density = True)

plt.hist(df['AvgHomeValue'],bins = 20, density = True)

#%%-错误值可以用缺失值替代，因为改为正确的值代价太大，所以用缺失值替代
df['AvgIncome'] = df['AvgIncome'].replace(0,np.NAN)
#-然后对缺失值进行补缺（缺失值<20%-连续变量使用均值或者中位数填补，分类变量不需要填补，算一类即可，或者用众数填补）
#-填补缺失值的同时可以生成一个虚拟变量，用于表示是否是进行了填补的缺失值。
vmean = df['Age'].mean(axis = 0,skipna= True)
df['age_empflag'] = df['Age'].isnull()
df['Age'] = df['Age'].fillna(vmean)


