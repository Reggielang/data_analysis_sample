# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:37:24 2020

@author: REGGIE
"""

#%%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt


os.chdir(r'C:\Users\REGGIE\Desktop\数据分析资料\280_Ben_八大直播八大案例配套课件\实例\data')

model_data = pd.read_csv("profile_telecom.csv")
data = model_data.drop('ID',axis =1)
data.head()

#%%
#查看相关系数矩阵，判定做变量降维的必要性（非必须）

corr_matrix = data.corr(method='pearson')
#corr_matrix = corr_matrix.abs()
corr_matrix
#根据相关系数矩阵，第三个属性和第四个属性相关系数比较高

#%%
#做主成分之前，进行中心标准化

from sklearn import preprocessing
data = preprocessing.scale(data)

#%%
#使用sklearn的主成分分析，用于判断保留主成分的数量
from sklearn.decomposition import PCA
'''
1、第一次的n_components参数应该设的大一点
2、观察explained_variance_ratio_和explained_variance_的取值变化，建议explained_variance_ratio_累积大于0.85，explained_variance_需要保留的最后一个主成分大于0.8，
'''
pca=PCA(n_components=4)
newData=pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

#通过主成分在每个变量上的权重的绝对值大小，确定每个主成分的代表性
pd.DataFrame(pca.components_).T
#%%
# 第二步：根据主成分分析确定需要保留的主成分数量，进行因子分析
# 导入包，并对输入的数据进行主成分提取。为保险起见，data需要进行中心标准化

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(
        data,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()
#%%
#设定提取主成分的方式。默认为“broken_stick”方法，建议使用“top_n”法

fa.find_comps_to_retain(method='top_n',num_keep=3)
#%%
# - 3、通过最大方差法进行因子旋转

fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)

# - 说明：可以通过第三张图观看每个因子在每个变量上的权重，权重越高，代表性越强
# - 4、获取因子得分
pd.DataFrame(fa.comps["rot"])

#%%
import numpy as np
fas = pd.DataFrame(fa.comps["rot"])
data = pd.DataFrame(data)
score = pd.DataFrame(np.dot(data, fas))
# 第三步：根据因子得分进行数据分析

fa_scores=score.rename(columns={0: "wei_web", 1: "call", 2: "msg"})
fa_scores.head()
#%%
# ### 第四步：使用因子得分进行k-means聚类
#4.1 k-means聚类的第一种方式：不进行变量分布的正态转换--用于寻找异常值
# 1、查看变量的偏度

var = ["wei_web","call","msg"]
skew_var = {}
for i in var:
    skew_var[i]=abs(fa_scores[i].skew())
    skew=pd.Series(skew_var).sort_values(ascending=False)
skew
#%%
# - 2、进行k-means聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) #MiniBatchKMeans()分批处理
#kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=1)
result=kmeans.fit(fa_scores)
#print(result)
#对分类结果进行解读

model_data_l=model_data.join(pd.DataFrame(result.labels_))
model_data_l=model_data_l.rename(columns={0: "clustor"})
model_data_l.head()


model_data_l.clustor.value_counts().plot(kind = 'pie')

#%%
# ### 4.2 k-means聚类的第二种方式：进行变量分布的正态转换--用于客户细分
# - 1、进行变量分布的正态转换

import numpy as np
from sklearn import preprocessing
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
fa_scores_trans=quantile_transformer.fit_transform(fa_scores)
fa_scores_trans=pd.DataFrame(fa_scores_trans)
fa_scores_trans=fa_scores_trans.rename(columns={0: "wei_web", 1: "call", 2: "msg"})
fa_scores_trans.head()

#%%
var = ["wei_web","call","msg"]
skew_var = {}
for i in var:
    skew_var[i]=abs(fa_scores_trans[i].skew())
    skew=pd.Series(skew_var).sort_values(ascending=False)
skew

#%%
# - 2、进行k-means聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4) #MiniBatchKMeans()分批处理
#kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=1)
result=kmeans.fit(fa_scores_trans)
#print(result)

# - 3、对分类结果进行解读
model_data_l=model_data.join(pd.DataFrame(result.labels_))
model_data_l=model_data_l.rename(columns={0: "clustor"})
model_data_l.head()

#%%
model_data_l.clustor.value_counts().plot(kind = 'pie') 

#%%
#导入决策树模型
data1 = model_data.loc[ :,'cnt_call':'cnt_web']

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=100, min_samples_leaf=100, random_state=12345)  # 当前支持计算信息增益和GINI
clf.fit(data1, result.labels_)

#%%
#查看用户分类，以及用户群体之间的不同特征
import pydotplus
from IPython.display import Image
import sklearn.tree as tree

dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=data1.columns,  
                                class_names=['0','1','2','3'],
                                filled=True) 

graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
 