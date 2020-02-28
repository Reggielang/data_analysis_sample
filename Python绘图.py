# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:57:59 2020

@author: REGGIE
"""
#%%(用于分区)
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
df = pd.read_csv('sndHsPr.csv')
#%%--一个变量的分析
district = {'fengtai':'丰台区', 'haidian': '海淀区',  'chaoyang': '朝阳区', 'dongcheng':'东城区', 'xicheng':'西城区', 'shijingshan':'石景山区'}
df['district'] = df.dist.map(district)
df.head()

#%%
df.district.value_counts()
#向量的绘图
df.district.value_counts().plot(kind='bar')
df.district.value_counts().plot(kind='pie')

#%%
df.price.mean()
df.price.median()
df.price.std()
df.price.skew()
df.price.agg(['mean','median','std','skew'])

#%%
df.price.hist(bins = 40)

#%%
#表示取这个数据的百分之多少
df.price.quantile([0.01,0.5,0.99])

#%%--两个变量的分析
sub_df = pd.crosstab(df.district,df.school)

#分类柱形图
pd.crosstab(df.dist,df.subway).plot(kind='bar')

#堆叠柱形图
sub_df.plot(kind='bar',stacked = True)

#%%
#标准化的堆叠图-1代表列，0代表行
sub_df = pd.crosstab(df.district,df.school)
sub_df['sum1'] = sub_df.sum(1)
sub_df = sub_df.div(sub_df.sum1,axis = 0)
sub_df[[0,1]].plot(kind = 'bar',stacked = True)

#%%
from stack2dim import *
stack2dim(df,i = 'district',j = 'school') 
