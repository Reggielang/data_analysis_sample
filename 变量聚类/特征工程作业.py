# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:50:18 2020

@author: REGGIE
"""
#%%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from woe import WoE

os.chdir(r'C:\Users\REGGIE\Desktop\数据分析资料\280_Ben_八大直播八大案例配套课件\实例')

df = pd.read_csv('CSR_CHURN_Samp.csv')

df.STA_DTE.value_counts()
#查看是否存在缺失值
df.isna().sum()
#查看后发现Edu存在缺失值
#使用均值填补法（让模型更稳定）
df['Edu'].value_counts()
df['Buy_Type'] .value_counts()
df['Edu'] = df['Edu'].apply(lambda x : x.replace('x','6'))
df['Buy_Type'] = df['Buy_Type'].apply(lambda x : x.replace('n','4'))
df = df.drop(['CSR_ID','STA_DTE'],axis = 1)
df[['Value','R3m_Avg_Cns_Amt','R1m_Trd3_Cns_Amt','Ctr_R1y','Ilt_Bal_Amt','Net_Cns_Cnt','Ovs_Cns_Amt','R6m_Avg_Rdm_Pts','R3m_Max_Csh_Amt','R6m_Max_Csh_Amt','R12m_Max_Csh_Amt']] = df[['Value','R3m_Avg_Cns_Amt','R1m_Trd3_Cns_Amt','Ilt_Bal_Amt','Ctr_R1y','Net_Cns_Cnt','Ovs_Cns_Amt','R6m_Avg_Rdm_Pts','R3m_Max_Csh_Amt','R6m_Max_Csh_Amt','R12m_Max_Csh_Amt']].applymap(lambda x: x.replace(',', ''))
df = df.astype('int')
df.info()
#%%
#对涉及的X进行分箱处理
y = 'Evt_Flg'
#连续变量
var_c = ['Value','Age','Csr_Dur','Ctr_R1y','R3m_Avg_Cns_Amt','R1m_Trd3_Cns_Amt','Ilt_Bal_Amt',
         'Ovs_Cns_Amt','R6m_Avg_Rdm_Pts','R3m_Max_Csh_Amt','R6m_Max_Csh_Amt','R12m_Max_Csh_Amt',
         'R3m_Max_Ilt_Amt','R6m_Max_Ilt_Amt','Net_Cns_Cnt']
#分类变量
var_d = ['Gen','Edu','Buy_Type','Ovs_Cns_Cnt','R12m_Avg_Cns_Cnt','R6m_Csh_Mth_Nbr',
         'Lmth_Fst_Ilt','Lmth_Fst_Int','Total_call_nbr','R6M_CALL_NBR','R6M_CLS_NBR']

#%%
#得到数据集X，以及目标Y
X = df[var_c + var_d].copy()
Y = df[y].copy()
#%%
## 筛选预测能力强的变量
#根据IV值筛选变量 - 分类变量
iv_d = {}
for i in var_d:
    iv_d[i] = WoE(v_type='d').fit(X[i].copy(), Y.copy()).iv

pd.Series(iv_d).sort_values(ascending = False)

#%%
# 保留iv值较高的分类变量
var_d_s = ['Lmth_Fst_Ilt', 'Total_call_nbr','R6M_CALL_NBR','R6m_Csh_Mth_Nbr','Ovs_Cns_Cnt','R12m_Avg_Cns_Cnt']

#%%
## 根据IV值筛选变量-连续变量
iv_c = {}
for i in var_c:
    iv_c[i] = WoE(v_type='c').fit(X[i],Y).iv 

sort_iv_c = pd.Series(iv_c).sort_values(ascending=False)
sort_iv_c
#%%
#根据日期分为训练数据集，测试数据集
df1 = df[df['STA_DTE'].isin(['31Jan2015'])]
df2 = df[df['STA_DTE'].isin(['31Dec2014'])]
#%%


#%%




