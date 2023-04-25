import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity #Bartlett’s球状检验
from factor_analyzer.factor_analyzer import calculate_kmo #KMO检验
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#导入数据
data=pd.read_csv('D:\\2022国赛数模练习题\\附件2.csv',encoding='gbk')
#剔除无用的列
data.drop(columns='编号',inplace=True)

#进行数据标准化操作
def standardization(data): #data是矩阵,主要功能是进行标准化,输出是经过标准化的矩阵
    data_std=[np.std(data[:,i]) for i in range(data.shape[1])]
    data_mean=[np.mean(data[:,i]) for i in range(data.shape[1])]
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-data_mean[i])/data_std[i]
    return data
data_columns=data.columns#存储列名
data=pd.DataFrame(standardization(data.values),columns=data_columns)

#下面进行充分性检验（下面两步骤在不严格的情况下可以忽略）
#1Bartlett’s球状检验
chi_square_value, p_value = calculate_bartlett_sphericity(data)
print(chi_square_value,p_value) #p值小于0.05，检验通过，能进行主成分分析

#2KMO检验
kmo_all,kmo_model=calculate_kmo(data)
print(kmo_model) #大于0.6说明变量之间的相关性强，能进行主成分分析

#下面运用sklearn中的PCA函数进行主成分分析建模
#先观察方差贡献率来选择因子数量
pca=PCA(n_components=8) #提取因子数量（先选多点）
pca.fit(data)
explained_variance=pca.explained_variance_ #贡献方差，即特征根
variance_ratio=pca.explained_variance_ratio_  #方差贡献率(已经从大到小排序)
score_matrix=pca.components_ #成分得分系数矩阵(载荷系数矩阵)

#下面进行绘制碎石图选取因子数量
'''plt.figure(figsize=(8,6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(1,data.values.shape[1]+1),variance_ratio,linestyle=':')
plt.scatter(range(1,data.values.shape[1]+1),variance_ratio,marker='o',)
plt.annotate('0.056',xy=(4,0.096),xytext=(4,0.15),bbox=dict(boxstyle='square',fc='firebrick'),
               arrowprops=dict(facecolor='steelblue',shrink=0.002),fontsize=9.5,color='white') #打上标签
plt.title('碎石图',fontsize=15)
plt.xlabel('因子个数',fontsize=13)
plt.ylabel('方差贡献率',fontsize=13)
plt.show()'''

#下面进行观测载荷系数矩阵（选取因子为4）
pca=PCA(n_components=4)
pca.fit(data)
score_matrix=pca.components_.T #原shape是(4,8)要进行转置一下，
data1=pd.DataFrame(score_matrix,index=data_columns) #注意这里要在index上给上标签（竖着的），热点图的规则

plt.figure(figsize = (8,6))
#进行绘制热力图
ax = sns.heatmap(data1, annot=True, cmap="BuPu")
# 设置y轴字体大小
ax.yaxis.set_tick_params(labelsize=10)
#设置标题
plt.title("PCA Loading Factor Matrix", fontsize="xx-large")
# 设置y轴标签
plt.ylabel("main ingredient", fontsize="13")
# 显示图片
plt.show()

#下面进行数据降维操作（最关键的）
trans_data=pca.transform(data)

#用于主成分分析法的辅助函数
def takefirst(elem):
    return elem[0]

#主成分分析法(自设计算法)
def pca(X,k): #X是待降维的输入矩阵，k是我们想要降维到的维数，输出是降维后的矩阵
    n_samples,n_features=X.shape #获取行数和列数
    mean=np.array([np.mean(X[:,i]) for i in range(n_features)]) #获取每一列的均值
    #进行标准化操作
    norm_X=X-mean
    #求原矩阵和其逆矩阵的点积
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
    #计算特征值和特征向量
    eig_val,eig_vec=np.linalg.eig(scatter_matrix)
    eig_pairs=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(n_features)] #将特征值和特征向量分别组合在一起
    eig_pairs.sort(key=takefirst,reverse=True) #按照特征值进行降序排序
    feature=np.array([ele[1] for ele in eig_pairs[:k]]) #选取前面K个特征向量
    result=np.dot(norm_X,np.transpose(feature))#进行转换
    return result

