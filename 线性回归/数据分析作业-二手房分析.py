# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:44:10 2020

@author: REGGIE
"""
###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from numpy import corrcoef,array
from statsmodels.formula.api import ols
import math
#修改字体显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

###
path = 'C:/Users/REGGIE/Desktop/数据分析资料/280_Ben_八大直播八大案例配套课件/实例/sechouse.csv'
df = pd.read_csv(path)
df.head()

district = {'fengtai':'丰台区', 
            'haidian': '海淀区',  
            'chaoyang': '朝阳区', 
            'dongcheng':'东城区', 
            'xicheng':'西城区', 
            'shijingshan':'石景山区'}
df['dist'] = df.dist.map(district)
df['price'] = df['price']/10000
df.describe()
df.describe().T

#因变量是右偏的，那么这种数据大多数情况都需要对Y值取对数
df.price.hist(bins = 100)
df.price.describe()

#整体来看(所有的自变量情况)
for i in range(7):
    if i != 3:
        print(df.columns.values[i],":")
        print(df[df.columns.values[i]].agg(['value_counts']).T)
        print("=======================================================================")
    else:
        continue
print('AREA:')
print(df.AREA.agg(['min','mean','median','max','std']).T)


#不同城区
df['dist'].value_counts()
df['dist'].value_counts().plot(kind = 'barh')
#roomnum
df['roomnum'].value_counts()
df['roomnum'].value_counts().plot(kind = 'bar')
#房屋客厅
df['halls'].value_counts()
df['halls'].value_counts().plot(kind = 'bar')
#房屋面积
df['AREA'].describe()
#房屋楼层
df['floor'].value_counts()
df['floor'].value_counts().plot(kind = 'bar')
#房屋是否临近地铁
df['subway'].value_counts()
df['subway'].value_counts().plot(kind = 'bar')
#房屋是否是学区房
df['school'].value_counts()
df['school'].value_counts().plot(kind = 'bar')



#自变量对因变量的影响
sns.boxplot(x='dist',y= 'price',data = df)

sns.barplot(x='roomnum', y='price',data = df)

sns.barplot(x='halls', y='price',data = df)


sns.barplot(x='floor',y= 'price',data = df)

sns.scatterplot(x = 'AREA', y = 'price',data= df)

sns.boxplot(x='subway',y= 'price',data = df)

sns.boxplot(x='school',y= 'price',data = df)

#看到从左至右逐渐稀疏的散点图,第一反应是对Y取对数
#
#房屋面积(取对数后)和单位面积房价（取对数后）的散点图
df['price_lg'] = np.log(df['price'])
df['AREA_lg'] = np.log(df['AREA'])
sns.scatterplot(x ='AREA_lg', y = 'price_lg',data= df,marker='.')

#求AREA_ln和price_ln的相关系数矩阵
#相关系数大于0.8是强相关，大于0.5小于0.8是中度相关，小于0.3是弱相关（基本不相关）
data1=array(df['price_lg'])
data2=array(df['AREA_lg'])
datB=array([data1,data2])
corrcoef(datB)


print(pd.crosstab(df.subway,df.school))
sub_sch=pd.crosstab(df.subway,df.school)
sub_sch = sub_sch.div(sub_sch.sum(1),axis = 0)
sub_sch.plot(kind = 'bar')

# # 2 建模

# In[38]:
#1、首先检验每个解释变量是否和被解释变量独立
#%%由于原始样本量太大，无法使用基于P值的构建模型的方案，因此按照区进行分层抽样
#dat0 = datall.sample(n=2000, random_state=1234).copy()
def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    """
    对输入的 dataframe 进行抽样的函数

    参数:
        - df: 输入的数据框 pandas.dataframe 对象

        - sampling:抽样方法 str
            可选值有 ["simple_random", "stratified", "systematic"]
            按顺序分别为: 简单随机抽样、分层抽样、系统抽样

        - k: 抽样个数或抽样比例 int or float
            (int, 则必须大于0; float, 则必须在区间(0,1)中)
            如果 0 < k < 1 , 则 k 表示抽样对于总体的比例
            如果 k >= 1 , 则 k 表示抽样的个数；当为分层抽样时，代表每层的样本量

        - stratified_col: 需要分层的列名的列表 list
            只有在分层抽样时才生效

    返回值:
        pandas.dataframe 对象, 抽样结果
    """
    import random
    import pandas as pd
    from functools import reduce
    import numpy as np
    import math
    
    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n=True
        if sampling is "stratified":
            alln=k*df.groupby(by=stratified_col)[stratified_col[0]].count().count() # 有问题的
            #alln=k*df[stratified_col].value_counts().count() 
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n=False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)
        
    #print(k)

    if sampling is "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        res_df = df.iloc[idx,:].copy()
        return res_df

    elif sampling is "systematic":
        print("使用系统抽样")
        step = len_df // k+1          #step=len_df//k-1
        start = 0                  #start=0
        idx = range(len_df)[start::step]  #idx=range(len_df+1)[start::step]
        res_df = df.iloc[idx,:].copy()
        #print("k=%d,step=%d,idx=%d"%(k,step,len(idx)))
        return res_df

    elif sampling is "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.in1d(stratified_col, df.columns)), "请检查输入的列名"
        
        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n==True:
            group_k = grouped.map(lambda x:k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))
        
        res_df = df.head(0)
        for df_idx in group_k.index:
            df1=df
            if len(stratified_col)==1:
                df1=df1[df1[stratified_col[0]]==df_idx]
            else:
                for i in range(len(df_idx)):
                    df1=df1[df1[stratified_col[i]]==df_idx[i]]
            idx = random.sample(range(len(df1)), group_k[df_idx])
            group_df = df1.iloc[idx,:].copy()
            res_df = res_df.append(group_df)
        return res_df

    else:
        raise AssertionError("sampling is illegal")


#得到分层的随机抽样的新数据集
data=get_sample(df, sampling="stratified", k=400, stratified_col=['dist'])
#%%
print("dist的P值为:{}".format(sm.stats.anova_lm(ols('price ~ C(dist)',data=data).fit())._values[0][4]))
print("roomnum的P值为:{}".format(sm.stats.anova_lm(ols('price ~ C(roomnum)',data=data).fit())._values[0][4]))#明显高于0.001->不显著->独立
print("halls的P值为:{}".format(sm.stats.anova_lm(ols('price ~ C(halls)',data=data).fit())._values[0][4]))#高于0.001->边际显著->暂时考虑
print("floor的P值为:{}".format(sm.stats.anova_lm(ols('price ~ C(floor)',data=data).fit())._values[0][4]))#高于0.001->边际显著->暂时考虑
print("subway的P值为:{}".format(sm.stats.anova_lm(ols('price ~ C(subway)',data=data).fit())._values[0][4]))
print("school的P值为:{}".format(sm.stats.anova_lm(ols('price ~ C(school)',data=data).fit())._values[0][4]))

#%%
###厅数不太显著，考虑做因子化处理，变成二分变量，使得建模有更好的解读
###将是否有厅bind到已有数据集
data['style_new']=data.halls
data.style_new[data.style_new>0]='有厅'
data.style_new[data.style_new==0]='无厅'
data.head()
#%%
#对于多分类变量，生成哑变量，并设置基准--完全可以在ols函数中使用C参数来处理虚拟变量
dum=pd.get_dummies(data[['dist','floor']])
dum.head()
#生成了哑变量数量应该是K-1个，所以要删除一些哑变量
dum.drop(['dist_石景山区','floor_high'],axis=1,inplace=True)#这两个是参照组-在线性回归中使用C函数也可以
dum.head()
#%%
#生成的哑变量与其他所需变量合并成新的数据框
data1=pd.concat([dum,data[['school','subway','style_new','roomnum','AREA','price']]],axis=1)
data1.head()
#%%
###线性回归模型
#lm1 = ols("price ~ dist_丰台区+dist_朝阳区+dist_东城区+dist_海淀区+dist_西城区+school+subway+floor_middle+floor_low+style_new+roomnum+AREA", data=dat1).fit()
lm1 = ols("price ~ dist_丰台区+dist_朝阳区+dist_东城区+dist_海淀区+dist_西城区+school+subway+floor_middle+floor_low+AREA", data=data1).fit()
lm1_summary = lm1.summary()
lm1_summary  #回归结果展示

#%%
lm2 = ols("price ~ C(dist)+school+subway+C(floor)+AREA", data=data).fit()
lm2_summary = lm2.summary()
lm2_summary  #回归结果展示



#%%
data1['pred1']=lm1.predict(data1)
data1['resid1']=lm1.resid
data1.plot('pred1','resid1',kind='scatter')  #模型诊断图，存在异方差现象，对因变量取对数
#%%
###对数线性模型，取对数之后，自变量的比较为百分比，比如，西城区比石景山区贵63%
data1['price_ln'] = np.log(data1['price'])  #对price取对数
data1['AREA_ln'] = np.log(data1['AREA'])#对AREA取对数

lm3 = ols("price_ln ~ dist_丰台区+dist_朝阳区+dist_东城区+dist_海淀区+dist_西城区+school+subway+floor_middle+floor_low+AREA", data=data1).fit()
lm3_summary = lm3.summary()
lm3_summary  #回归结果展示

#%%
lm4 = ols("price_ln ~ dist_丰台区+dist_朝阳区+dist_东城区+dist_海淀区+dist_西城区+school+subway+floor_middle+floor_low+AREA_ln", data=data1).fit()
lm4_summary = lm4.summary()
lm4_summary  #回归结果展示

#%%
###假想情形，做预测，x_new是新的自变量
x_new1=data1.head(1)
x_new1
#%%
x_new1['dist_朝阳区']=0

x_new1['dist_东城区']=1
x_new1['roomnum']=2
x_new1['halls']=1
x_new1['AREA_ln']=np.log(70)
x_new1['subway']=1
x_new1['school']=1
x_new1['style_new']="有厅"

#预测值 保留2位小数
print("单位面积房价：",round(math.exp(lm4.predict(x_new1)),2),"万元/平方米")
print("总价：",round(math.exp(lm4.predict(x_new1))*70,2),"万元")
#那么改变X_new中的一些值，满足作业条件，经过预测 需要总价541万



#def main():
    #search()


#if __name__ == '__main__':
   # main()   