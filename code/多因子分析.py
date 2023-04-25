import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity #Bartlett’s球状检验
from factor_analyzer.factor_analyzer import calculate_kmo #KMO检验

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

#下面进行充分性检验
#1Bartlett’s球状检验
chi_square_value, p_value = calculate_bartlett_sphericity(data)
print('Bartlett球状检验为：',chi_square_value,p_value) #p值小于0.05，检验通过，能进行因子分析

#2KMO检验
kmo_all,kmo_model=calculate_kmo(data)
print('KMO检验为：',kmo_model) #大于0.6说明变量之间的相关性强，能进行因子分析

#下面进行选择因子数量，先计算变量的特征值和特征向量
faa=FactorAnalyzer(8,rotation=None)
faa.fit(data) #将数据导入模型中

#得到特征值ev和特征向量v
ev,v=faa.get_eigenvalues()

#下面进行绘制碎石图(绘制特征值和因子个数的变化) 经过观察此时可以选择3个因子
plt.figure(figsize=(8,6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(1,data.values.shape[1]+1),ev,linestyle=':')
plt.scatter(range(1,data.values.shape[1]+1),ev,marker='o',)
plt.annotate('0.767',xy=(4,0.767),xytext=(4.3,1),bbox=dict(boxstyle='square',fc='firebrick'),
               arrowprops=dict(facecolor='steelblue',shrink=0.002),fontsize=9.5,color='white') #打上标签
plt.title('碎石图',fontsize=15)
plt.xlabel('因子个数',fontsize=13)
plt.ylabel('特征值',fontsize=13)
plt.show()

#下面选择3个因子来进行因子分析的建模过程同时指定矩阵旋转方式为：方差最大化
faa_three=FactorAnalyzer(3,rotation='varimax') #构建模型并设置参数
faa_three.fit(data) #将数据导入到模型中

#下面进行查看因子贡献率（三种1总方差贡献2方差贡献率3累积方差贡献率）
variance_dataframe=pd.DataFrame(faa_three.get_factor_variance(),columns=['feature1','feature2','feature3'])
print(variance_dataframe)

#下面进行绘制各个降维特征的方差贡献率和累计方差贡献率
width=0.25 #柱形宽度
index=np.arange(variance_dataframe.values.shape[1]) #序号
feature_index=['feature1','feature2','feature3'] #最总显示的降维特征名称
plt.rc('font',family='Times New Roman')  #更改画图字体为Times New Roman
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
#下面进行绘制累计方差贡献率曲线
plt.plot(index,variance_dataframe.iloc[2,:],marker='o',color='r',alpha=0.5,lw=2,label='Contribution rate curve of cumulative variance')
#下面进行绘制各公共因子的方差贡献率柱形图
plt.bar(index,variance_dataframe.iloc[1,:],width=width, alpha=0.4,color='b' ,label='Variance contribution rate')
#下面进行打上数据标签
for i,data1 in enumerate(variance_dataframe.iloc[1,:]): #方差贡献率
    plt.text(i,data1+0.004,round(data1,1),horizontalalignment='center',fontsize=12)
for i,data1 in enumerate(variance_dataframe.iloc[2,:]): #累计方差贡献率
    if i!=0: #第一个不打标签，防止重复
        plt.text(i,data1+0.004,round(data1,1),horizontalalignment='center',fontsize=12)
plt.xticks(index,feature_index) #打上x轴坐标标签
plt.xlabel('Feature type',fontsize=13)
plt.ylabel('contribution',fontsize=13)
plt.title('Contribution rate of each characteristic variance and cumulative contribution rate',fontsize=15)
plt.legend()
plt.show()

#下面可以计算系数矩阵并绘制出热力图（最后进行观察分析即可，观察每一列看谁的系数大可以将几个变量分为一组，每个因子都可以分一组）
data1=pd.DataFrame(faa_three.loadings_,index=data_columns)
plt.figure(figsize = (8,6))
#进行绘制热力图
ax = sns.heatmap(data1, annot=True, cmap="BuPu") #注意这里要在index上给上标签（竖着的），热点图的规则
# 设置y轴字体大小
ax.yaxis.set_tick_params(labelsize=10)
#设置标题
plt.title("Factor Analysis", fontsize="xx-large")
# 设置y轴标签
plt.ylabel("Sepal Width", fontsize="13")
# 显示图片
plt.show()
#plr.savefig()

#最后还可以将数据降维（转化）
trans_data=faa_three.transform(data)
print(trans_data)



