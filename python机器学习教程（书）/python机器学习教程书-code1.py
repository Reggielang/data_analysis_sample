# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:42:25 2020

@author: REGGIE
"""
#%%
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(
        iris_dataset['data'],iris_dataset['target'], random_state = 0)

#%%散点图矩阵，可以两两查看所有的特征。
from pandas.plotting import scatter_matrix
import mglearn
# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
 hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


#%% K 近邻分类模型

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

#%% 使用模型做出一次预测
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape)) # 将这朵花的测量数据转换为二维 NumPy 数组的一行，这是因为 scikit-learn 的输入数据必须是二维数组。

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
 iris_dataset['target_names'][prediction]))

#%% 评估模型
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {}".format(np.mean(y_pred == y_test)))

print("Test set score: {}".format(knn.score(X_test, y_test)))