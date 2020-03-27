
# coding: utf-8

# # 逻辑回归
#subscriberID="个人客户的ID"
#churn="是否流失：1=流失";
#Age="年龄"
#incomeCode="用户居住区域平均收入的代码"
#duration="在网时长"
#peakMinAv="统计期间内最高单月通话时长"
#peakMinDiff="统计期间结束月份与开始月份相比通话时长增加数量"
#posTrend="该用户通话时长是否呈现出上升态势：是=1"
#negTrend="该用户通话时长是否呈现出下降态势：是=1"
#nrProm="电话公司营销的数量"
#prom="最近一个月是否被营销过：是=1"
#curPlan="统计时间开始时套餐类型：1=最高通过200分钟；2=300分钟；3=350分钟；4=500分钟"
#avPlan="统计期间内平均套餐类型"
#planChange="统计结束时和开始时套餐的变化：正值代表套餐档次提升，负值代表下降，0代表不变"
#posPlanChange="统计期间是否提高套餐：1=是"
#negPlanChange="统计期间是否降低套餐：1=是"
#call_10086="拨打10086的次数"


# In[1]:

import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#pd.set_option('display.max_columns', None)
os.chdir(r"D:\Python_Training\script_Python\8logistic\HW")

# 导入数据和数据清洗

# In[5]:

churn = pd.read_csv(r'telecom_churn.csv', skipinitialspace=True)
churn.head()

#1两变量分析：检验该用户通话时长是否呈现出上升态势(posTrend)对流失(churn) 是否有预测价值
# ##  分类变量的相关关系
#
# 交叉表

# In[6]:

cross_table = pd.crosstab(churn.posTrend, 
                         churn.churn, margins=True)
cross_table


# 列联表

# In[7]:

def percConvert(ser):
    return ser/float(ser[-1])

cross_table.apply(percConvert, axis=1)


# In[8]:

print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(cross_table.iloc[:2, :2]))

#2首先将原始数据拆分为训练和测试数据集，使用训练数据集建立在网时长对流失的逻辑回归，使用测试数据集制作混淆矩阵
#（阈值为0.5），提供准确性、召回率指标，提供ROC曲线和AUC。
# ## 逻辑回归

# In[9]:

churn.plot(x='duration', y='churn', kind='scatter')


# •随机抽样，建立训练集与测试集

# In[10]:

train = churn.sample(frac=0.7, random_state=1234).copy()
test = churn[~ churn.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))


# In[11]:

lg = smf.glm('churn ~ duration', data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg.summary()

# 预测

# In[19]:

train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)

test['proba'].head()

# In[12]:
# ## 模型评估
# 
# 设定阈值

# In[20]:

test['prediction'] = (test['proba'] > 0.5).astype('int')


# 混淆矩阵

# In[22]:

pd.crosstab(test.churn, test.prediction, margins=True)


# - 计算准确率

# In[23]:

acc = sum(test['prediction'] == test['churn']) /np.float(len(test))
print('The accurancy is %.2f' %acc)


# In[25]:
"""
for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.churn,
                                   margins = True)
    precision = confusion_matrix.ix[0, 0] /confusion_matrix.ix['All', 0]
    recall = confusion_matrix.ix[0, 0] / confusion_matrix.ix[0, 'All']
    Specificity = confusion_matrix.ix[1, 1] /confusion_matrix.ix[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))

"""
# - 绘制ROC曲线

# In[27]:

import sklearn.metrics as metrics

fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()


# In[28]:

print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

#3使用向前逐步法从其它备选变量中选择变量，构建基于AIC的最优模型，绘制ROC曲线，同时检验模型的膨胀系数。
# In[14]:
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


# In[16]:

candidates = ['churn','duration','AGE','edu_class','posTrend','negTrend','nrProm','prom','curPlan','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]

lg_m1 = forward_select(data=data_for_select, response='churn')
lg_m1.summary()


# Seemingly wrong when using 'statsmmodels.stats.outliers_influence.variance_inflation_factor'

# In[17]:

def vif(df, col_i):
    from statsmodels.formula.api import ols
    
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)


# In[18]:

exog = train[candidates].drop(['churn'], axis=1)

for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))
#posTrend,negTrend;curPlan,avgplan有明显的共线性问题,剔除其中两个后重新建模.
#%%
#4）使用岭回归和Laso算法重建第三步中的模型，使用交叉验证法确定惩罚参数(C值)。并比较步骤四中Laso算法得到的模型和第三步得到的模型的差异
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

candidates = ['duration','AGE','edu_class','posTrend','negTrend','nrProm','prom','curPlan','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
#data_for_select = churn[candidates]
scaler = StandardScaler()  # 标准化
X = scaler.fit_transform(churn[candidates])
y = churn['churn']


#%%
from sklearn import linear_model
from sklearn.svm import l1_min_c
cs = l1_min_c(X, y, loss='log') * np.logspace(0, 4)


print("Computing regularization path ...")
#start = datetime.now()
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
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
#%%
cs = l1_min_c(X, y, loss='log') * np.logspace(0, 4)
import matplotlib.pyplot as plt #可视化模块  
from sklearn.cross_validation import cross_val_score # K折交叉验证模块  
  
k_scores = []  
clf = linear_model.LogisticRegression(penalty='l1')
#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率  
for c in cs:  
    clf.set_params(C=c)
    scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')  #http://scikit-learn.org/stable/modules/model_evaluation.html
    k_scores.append([c,scores.mean(),scores.std()])  
#%%  
#可视化数据  
#%%
data=pd.DataFrame(k_scores)#将字典转换成为数据框
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(np.log10(data[0]), data[1],'b')
ax1.set_ylabel('Mean ROC(Blue)')
ax1.set_xlabel('log10(cs)')
ax2 = ax1.twinx()
ax2.plot(np.log10(data[0]), data[2],'r')
ax2.set_ylabel('Std ROC Index(Red)')

#得到合理的C为 np.exp(-1.9)
#%%
#重新实现Laso算法
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

candidates = ['duration','AGE','edu_class','posTrend','negTrend','nrProm','prom','curPlan','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
#data_for_select = churn[candidates]
scaler = StandardScaler()  # 标准化
X = scaler.fit_transform(churn[candidates])
y = churn['churn']


#%%
from sklearn import linear_model

clf = linear_model.LogisticRegression(C=np.exp(-1.9), penalty='l1')
clf.fit(X, y)
clf.coef_
#%%




#以下是KNN算法
"""
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法  
k_range = range(1, 40)  
  
k_scores = []  
  
#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率  
for k in k_range:  
    knn = KNeighborsClassifier(n_neighbors=k)  
    scores = cross_val_score(knn, X, y, cv=3, scoring='roc_auc')  
    k_scores.append([k,scores.mean(),scores.std()])  
#%%  
#可视化数据  
plt.plot(k_range, k_scores[1])  
plt.xlabel('Value of K for KNN')  
plt.ylabel('Cross-Validated Accuracy')  
plt.show()  
"""
#%%