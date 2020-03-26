# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:24:43 2020

@author: REGGIE
"""

tree.export_graphviz(clf, out_file='cart.dot')


# 可以使用graphviz将树结构输出，在python中嵌入graphviz可参考：[pygraphviz](http://www.blogjava.net/huaoguo/archive/2012/12/21/393307.html)

# # 可视化

# 使用dot文件进行决策树可视化需要安装一些工具：
# - 第一步是安装graphviz。linux可以用apt-get或者yum的方法安装。如果是windows，就在官网下载msi文件安装。
#    无论是linux还是windows，装完后都要设置环境变量，将graphviz的bin目录加到PATH，
#    比如windows，将C:/Program Files (x86)/Graphviz2.38/bin/加入了PATH
# - 第二步是安装python插件graphviz： pip install graphviz
# - 第三步是安装python插件pydotplus: pip install pydotplus

# In[11]:


import pydotplus
from IPython.display import Image
import sklearn.tree as tree


# In[18]:


dot_data = tree.export_graphviz(
    clf, 
    out_file=None, 
    feature_names=data.columns,
    max_depth=5,
    class_names=['0','1'],
    filled=True
) 
            
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
#%%