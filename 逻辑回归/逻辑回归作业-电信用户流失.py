# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:11:21 2020

@author: REGGIE
"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

os.chdir(r'C:\Users\REGGIE\Desktop\数据分析资料\280_Ben_八大直播八大案例配套课件\实例')

df = pd.read_csv('telecom_churn.csv')

#%%
# 首先进行自变量与因变量之间的关系
df.columns
df.churn.value_counts()
#分类变量与分类变量
sns.boxplot(x = 'gender', y = 'churn',data = df)
#%%
#分类变量与连续变量
plt.figure(figsize=(20,5))

df.churn.groupby(df.AGE).sum().plot(kind = 'bar')

#%%
#分类变量与分类变量
#交叉表
pd.crosstab(df.edu_class,df.churn).plot(kind = 'bar')

#%%
#两变量分析：检验该用户通话时长是否呈现出上升态势(posTrend)对流失(churn) 是否有预测价值
# ##  分类变量的相关关系
#
# 交叉表

cross_table = pd.crosstab(df.posTrend, df.churn, margins=True)
cross_table
#呈现上升趋势的时候流失客户较少，所以有预测价值


#%%
#随机抽样，建立训练集与测试集

train = df.sample(frac=0.7, random_state=1234).copy()
test = df[~ df.index.isin(train.index)].copy()
print('训练集样本量: {0}, 测试集样本量: {1}'.format(len(train), len(test)))
#%%
#建立模型
lg = smf.glm('churn ~ duration', data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg.summary()

#%%
#进行训练和测试数据集的预测
train['predict'] = lg.predict(train)
test['predict'] = lg.predict(test)

test['predict'].head()

#%%
#ROC曲线

import sklearn.metrics as metrics

fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.predict)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.predict)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()
#%%
#使用向前逐步法从其它备选变量中选择变量，构建基于AIC的最优模型，绘制ROC曲线，同时检验模型的膨胀系数
#- 多元逻辑回归
# 向前法
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)


candidates = ['churn','duration','AGE','edu_class','posTrend','negTrend','nrProm','prom','curPlan','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]

lg_m1 = forward_select(data=data_for_select, response='churn')
lg_m1.summary()

#%%
def vif(df, col_i):
    from statsmodels.formula.api import ols
    
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

exog = train[candidates].drop(['churn'], axis=1)

for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))
 
#vif 大于10，并且两个变量之间的vif接近就代表具有强共线性
#posTrend,negTrend;curPlan,avgplan,nrProm,prom有明显的共线性问题,剔除他们具有强共线性中的一个重新建模.
#%%
final_data = ['churn','duration','AGE','edu_class','posTrend','nrProm','curPlan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[final_data]

lg_m2 = forward_select(data=data_for_select, response='churn')
lg_m2.summary()
#%%
#再次检验共线性问题
exog = train[final_data].drop(['churn'], axis=1)

for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))
#模型最优，不存在共线性，并且AIC已经最低

    
#%%
#2.机器学习中的回归
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

candidates = ['duration','AGE','edu_class','posTrend','negTrend','nrProm','prom','curPlan','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
#data_for_select = churn[candidates]
scaler = StandardScaler()  # 标准化
X = scaler.fit_transform(df[candidates])
y = df['churn']
    
#%%
#建立模型
from sklearn import linear_model
from sklearn.svm import l1_min_c
cs = l1_min_c(X, y, loss='log') * np.logspace(0, 4)


print("Computing regularization path ...")
#start = datetime.now()
clf = linear_model.LogisticRegression(C=1.0, tol=1e-6)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X, y)
    coefs_.append(clf.coef_.ravel().copy())
#print("This took ", datetime.now() - start)

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_)
ymin, ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()    

 