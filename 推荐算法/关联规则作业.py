# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:58:26 2020

@author: REGGIE
"""
#%%%
import pandas as pd
#from Apriori import *
import Apriori as apri
import matplotlib.pyplot as plt
import os 


os.chdir(r'C:\Users\REGGIE\Documents\GitHub\data_analysis_sample\data')
inverted=pd.read_csv('Prod.csv')
inverted.head()

#%%
#数据转换为相应的二维列表数据
idataset=apri.dataconvert(inverted,tidvar='ID',itemvar='PROD',data_type = 'inverted')
idataset[:5]

#%%
#关联规则
# 参数说明:
# + minSupport:最小支持度阈值
# + minConf:最小置信度阈值
# + minlen:规则最小长度
# + maxlen:规则最大长度
# 这里，minSupport或minConf设定越低，产生的规则越多，计算量也就越大
# 设定参数为:minSupport=0.05,minConf=0.5,minlen=1,maxlen=10

res = apri.arules(idataset,minSupport=0.01,minConf=0.1,minlen=1,maxlen=2)

#%%
# ## 产生关联规则
# + 规定提升度要大于1,并按照置信度进行排序


res.ix[res.lift>1,:].sort_values('support',ascending=False).head(20)
#%%
#互补品
res.ix[res.lift>1,['lhs','rhs','lift']].sort_values('lift',ascending=False).head(20)
#%%
#互斥品
res.ix[res.lift<1,['lhs','rhs','lift']].sort_values('lift',ascending=True).head(20)

#%%
#飞信这个产品和哪个产品之间互补性最强？如何设计捆绑销售？
feixin=res.loc[res.lhs==frozenset({'飞信'}),:]

#%%
#一个客户开通了手机报，根据关联规则，最该推荐哪三个产品？
bao=res.loc[res.lhs==frozenset({'手机报'}),:]

#%%
#公司决定推广咪咕音乐，需要向订购哪些产品的客户营销最好？
yinyue=res.loc[res.rhs==frozenset({'咪咕音乐'}),:]
