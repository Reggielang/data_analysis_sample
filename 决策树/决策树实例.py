# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:56:40 2020

@author: REGGIE
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns


os.chdir(r'C:\Users\REGGIE\Desktop\数据分析资料\280_Ben_八大直播八大案例配套课件\实例')

df = pd.read_csv('date_data2.csv')
df.describe()
#检查是否存在缺失值
df.isnull().sum()
#%%
#建模前的处理
#1.拆分训练数据，测试数据以及X,Y
import sklearn.model_selection as cross_validation

target = df['Dated']
data = df.iloc[:,:-1]

train_data, test_data, train_target, test_target = cross_validation.train_test_split(data,target, test_size=0.4, train_size=0.6 ,random_state=12345) # 划分训练集和测试集
#%%
# 选择决策树进行建模
import sklearn.tree as tree

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=6,min_samples_split=5)
clf.fit(train_data,train_target)

#%%
# 查看模型预测结果
train_est = clf.predict(train_data)  #  用模型预测训练集的结果
train_est_p=clf.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est = clf.predict(test_data) #  用模型预测测试集的结果
test_est_p=clf.predict_proba(test_data)[:,1] #  用模型预测测试集的概率
result = pd.DataFrame({'test_target':test_target,'test_est':test_est,'test_est_p':test_est_p}).T # 查看测试集预测结果与真实结果对比

#%%
#模型评估
import sklearn.metrics as metrics
print(metrics.confusion_matrix(test_target,test_est, labels = [0,1])) #混淆矩阵
print(metrics.classification_report(test_target,test_est)) #计算评估指标
print(pd.DataFrame(list(zip(data.columns,clf.feature_importances_)))) #每个变量的重要性指标


#%%
#查看关键的ROC曲线-分类器性能
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color='blue',label ='test')
plt.plot(fpr_train, tpr_train, color='red',label ='train')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#%%
#过度拟合，所以进行参数调优
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[2,3,4,5,6,7,8],
    'min_samples_split':[2,4,8,14,20,24,28] 
}
clf = tree.DecisionTreeClassifier()
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, 
                   scoring='roc_auc', cv=4)
clfcv.fit(train_data, train_target)
#%%
# 查看模型预测结果
train_est = clfcv.predict(train_data)  #  用模型预测训练集的结果
train_est_p=clfcv.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est=clfcv.predict(test_data)  #  用模型预测测试集的结果
test_est_p=clfcv.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率
result_2 = pd.DataFrame({'test_target':test_target,'test_est':test_est,'test_est_p':test_est_p}).T # 查看测试集预测结果与真实结果对比

#%%
# 查看新模型的预测结果
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color='blue')
plt.plot(fpr_train, tpr_train, color='red')
plt.title('ROC curve')
plt.show()

#%%
#查看最好的参数选择
clfcv.best_params_

#%%
#使用最好的参数进行建模
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=2) # 当前支持计算信息增益和GINI
clf.fit(train_data, train_target)  #  使用训练数据建模

#%%
# 决策树的可视化
import pydotplus
from IPython.display import Image
import sklearn.tree as tree

dot_data = tree.export_graphviz(
    clf, 
    out_file=None, 
    feature_names=train_data.columns,
    max_depth=5,
    class_names=['0','1'],
    filled=True
) 
            
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 

#%%
# 模型的保存于读取
import pickle as pickle

model_file = open(r'clf.model', 'wb')
pickle.dump(clf, model_file)
model_file.close()

model_load_file = open(r'clf.model', 'rb')
model_load = pickle.load(model_load_file)
model_load_file.close()

test_est_load = model_load.predict(test_data)
final = pd.crosstab(test_est_load,test_est)


#%%
#2.神经网络模型


df2 = pd.read_csv('date_data2.csv')
df2

#%%
## 划分训练集和测试集
from sklearn.model_selection import train_test_split
target2 = df2['Dated']
data2 = df2.iloc[:, :-1]

train_data2, test_data2, train_target2, test_target2 = train_test_split(
    data2, target2, test_size=0.4, train_size=0.6, random_state=1234) 

# 极差标准化-做神经网络之前，必须将数据进行标准化！！！而且必须采用极大极小值法！！！
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_data2)

scaled_train_data = scaler.transform(train_data2)
scaled_test_data = scaler.transform(test_data2)
#%%
# 然后进行建模

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='logistic', alpha=0.1, max_iter=1000)

mlp.fit(scaled_train_data, train_target2)
mlp
#%%
## 预测分类标签

train_predict = mlp.predict(scaled_train_data)

test_predict = mlp.predict(scaled_test_data)

#%%
# 预测概率

# 计算分别属于各类的概率，取标签为1的概率
train_proba = mlp.predict_proba(scaled_train_data)[:, 1]  
test_proba = mlp.predict_proba(scaled_test_data)[:, 1]

#%%
#验证模型好坏程度

from sklearn import metrics

print(metrics.confusion_matrix(test_target, test_predict, labels=[0, 1]))
print(metrics.classification_report(test_target, test_predict))


#%%
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target2, test_proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target2, train_proba)

plt.figure(figsize=[4, 4])
plt.plot(fpr_test, tpr_test, color = 'b')
plt.plot(fpr_train, tpr_train, color = 'r')
plt.title('ROC curve')
plt.show()

print('AUC = %6.4f' %metrics.auc(fpr_test, tpr_test))


#%%
#进行参数调优

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = {
    'hidden_layer_sizes':[(10, ), (15, ), (20, ), (5, 5)],
    'activation':['logistic', 'tanh', 'relu'], 
    'alpha':[0.001, 0.01, 0.1, 0.2, 0.4, 1, 10]
}
mlp = MLPClassifier(max_iter=1000)
gcv = GridSearchCV(estimator=mlp, param_grid=param_grid, 
                   scoring='roc_auc', cv=4, n_jobs=-1)
gcv.fit(scaled_train_data, train_target2)

#%%
gcv.best_score_

#%%
gcv.best_params_

#%%
#根据最有参数再次进行建模
from sklearn.neural_network import MLPClassifier

mlp2 = MLPClassifier(hidden_layer_sizes=(5,5), 
                    activation='tanh', alpha=0.01)

mlp2.fit(scaled_train_data, train_target2)
mlp2

#%%
#然后进行预测
final_perdict = mlp2.predict(scaled_test_data)


