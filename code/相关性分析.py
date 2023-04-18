import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from collections import Counter #拥有计数功能
from sklearn import metrics

#下面进行读取数据
data=pd.read_csv('C:\\Users\\86137\\Desktop\\2022数模国赛\\支撑材料\\data\\预测后的数据.csv',encoding='gbk')
data['颜色']=data['颜色'].fillna('未知') #填补缺失值
data_K=data[data['类型']=='高钾']
data_B=data[data['类型']=='铅钡']
chemical_columns=data.columns[5:]

#下面是先进行正态性检验
p_value_ks=[] #存储F检验的p值，p值大于0.05说明符合正态分布
for column in chemical_columns:
    mean=np.mean(data[column]) #计算每一列数据的均值
    std=np.std(data[column])#计算每一列数据的标准差
    result=stats.kstest(data[column],'norm',(mean,std))#主要的正态分布检验代码，p值大于0.05说明其符合正态分布
    p_value_ks.append(result[0])

#下面进行绘制pair图观察数据的分布状况(看是否符合线性相关的分布规律)
"""plt.figure(figsize=(8,6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.pairplot(data, hue='类型', aspect=1.5) #注意hue是分类变量（其可以看不同类别下的各特征之间的关系分布）
plt.show()"""

#下面进行pearson相关系数计算与可视化（如果存在线性相关性同时数据呈现正态分布的话就可以用）
#计算并绘制相关性图函数
def corr_heatmap1(data,title): #注意其中data是一个输入的dataframe,注意行名和列名哦都要改,title是图片的名称
    corrmat_data=data.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corrmat_data,annot=True,cmap='Purples') #cmap=Greens、Blues、Reds、Purples 也比较好看
    plt.title(title,fontsize=15)
    #plt.savefig(title + '.svg', format='svg', bbox_inches='tight')
    plt.show()
corr_heatmap1(data_K.iloc[5:],'K')
corr_heatmap1(data_B.iloc[5:],'B')
#皮尔森相关性系数检验
a1=stats.pearsonr(data_K['二氧化硅(SiO2)'],data_K['氧化钠(Na2O)']) #计算显著性检验p值,输出的第一个数是相关性系数，第二个数是p值

#下面进行斯皮尔曼相关性系数分析（如果数据呈现非正态分布，同时想要看其是否是单调相关的时候）
def corr_heatmap2(data,title): #注意其中data是一个输入的dataframe,注意行名和列名哦都要改,title是图片的名称
    corrmat_data=data.corr(method='spearman')
    plt.figure(figsize=(8,6))
    sns.heatmap(corrmat_data,annot=True,cmap='Greens') #cmap=Greens 也比较好看
    plt.title(title,fontsize=15)
    #plt.savefig(title + '.svg', format='svg', bbox_inches='tight')
    plt.show()

corr_heatmap2(data_K.iloc[5:],'K')
corr_heatmap2(data_B.iloc[5:],'B')
#斯皮尔曼相关性系数检验
a2=stats.spearmanr(data_K['二氧化硅(SiO2)'],data_K['氧化钠(Na2O)']) #计算显著性检验p值,输出的第一个数是相关性系数，第二个数是p值

#下面进行计算信息熵互信息来判断数据之间是否存在相关性（取值从 0~1）（适合应用于分类变量）直接可以使用sklearn中的函数进行实现
x=data['表面风化']
result=[]
for column in data.columns[1:4]: #注意原始数据不能有缺失值
    result.append(metrics.normalized_mutual_info_score(x, data[column])) #直接计算x与y的互信息，越大表示相关性越强
print("result_NMI:",result) #注意最后可以试试绘制热点图进行可视化表示

'''def Entropy(data):
    count=len(data) #计算总数量
    counter=Counter(data) #统计每个变量出现的次数，会是一个字典的形式返回
    prob={i[0]:i[1]/count for i in counter.items()} #这里先对counter字典中把关键字和值取出，同时计算每个概率p
    H=np.sum([i[1]*np.log2(i[1]) for i in prob.items()]) #这里进行计算信息熵（求和）
    return H

x=data['表面风化']
y=data['类型']
xy=list(zip(x,y)) #用于接下来计算联合分布概率
Hx=Entropy(x) #x的信息熵
Hy=Entropy(y) #y的信息熵
Hxy=Entropy(xy) #计算联合熵xy
H_x_y=Hxy-Hy #条件熵  X｜Y
H_y_x=Hxy-Hx # 条件熵  Y｜X'''





