import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer #第一个是标准化，第二个是归一化，第三个是向量归一化
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#导入数据
data=pd.read_csv('D:\\2022国赛数模练习题\\筛选后的白葡萄理化指标.csv',encoding='gbk')
target_value=data.iloc[:,-1] #假设最后一个特征是因变量
data=data.iloc[:,:-1]
data.drop(columns='样品编号',inplace=True)
data_columns=data.columns #29种自变量特征
#下面是三种类型的数据无量纲化方式
stand_scaler=StandardScaler()
Minmax_scaler=MinMaxScaler()
Normalizer_scaler=Normalizer()

stand_data=stand_scaler.fit_transform(data.values) #标准化后的原始数据(矩阵形式)
Minmax_data=Minmax_scaler.fit_transform(data.values) #归一化后的数据（矩阵形式）

#下面进行筛选特征
#第一种方法：方差筛选特征法(适用于量纲相同的，但是不能是经过标准化后的数据,可以用归一化后的数据)
Variance_list=np.std(Minmax_data,axis=0)
sort_list1=sorted(zip(data_columns.tolist(),Variance_list),key=lambda x: x[1],reverse=True) #按照方差值进行降序排序，再定阈值进行筛选即可

#第二种方式：相关系数法（Pearson相关系数、斯皮尔曼相关系数）
corr_data=pd.DataFrame(stand_data).corr()
corr_list=np.abs(corr_data.iloc[:,-1]) #绝对值相关系数
sort_list2=sorted(zip(data_columns.tolist(),corr_list),key=lambda x: x[1],reverse=True) #按照相关系数绝对值进行降序排序，再定阈值进行筛选即可

#第三种方式：基于树模型的特征选择法（随机森林算法、GBDT）
#1随机森林算法进行特征筛选
#先利用所有数据训练一个随机森林回归模型
RF=RandomForestRegressor(n_estimators=10,random_state=42)#构建随机森林模型，树的数量根据数据量和维数进行选择
RF.fit(stand_data,target_value.tolist()) #用所有数据训练模型
importances1=RF.feature_importances_ #输出特征重要性评分
sort_list3=sorted(zip(data_columns.tolist(),importances1.tolist()),key=lambda x:x[1],reverse=True) #按重要性评分从小到大进行排序

#2GBDT算法继续宁特征筛选
GBDT=GradientBoostingRegressor(n_estimators=10,random_state=42)
GBDT.fit(stand_data,target_value.tolist())
importances2=GBDT.feature_importances_
sort_list4=sorted(zip(data_columns.tolist(),importances2.tolist()),key=lambda x:x[1],reverse=True) #按重要性评分从小到大进行排序

#第4种方式：递归特征筛除法(不断剔除重要性最低的特征，并以最终模型的交叉验证精度来评估选择的特征)，注意其可以写伪代码
data_columns=data_columns.tolist()
cross_score_list=[] #存储不同数量的特征交叉验证的平均精度
for i in range(10): #这里可以设定最终剔除多少个特征
    RF=RandomForestRegressor(10,random_state=42)
    RF.fit(stand_data,target_value.tolist())
    #下面进行存储每个组合特征的交叉验证精度
    cross_score_list.append(np.abs(np.mean(cross_val_score(RF,stand_data,target_value.tolist(),cv=5,scoring='neg_mean_squared_error')))) #5折的交叉验证,使用的指标函数为R方'r2'，或者mse注意利用mse这种越小越好的指标时，sklearn中计算出的会带上负号
    #下面进行计算各特征重要性，并降序
    importances3=RF.feature_importances_
    sort_list5=sorted(zip(data_columns,importances3),key=lambda x:x[1],reverse=True)
    #去除重要性评分最低的特征(最后一个)
    drop_feature_index=data_columns.index(str(sort_list5[-1][0])) #得到要剔除特征的序号
    #data_columns=data_columns
    data_columns.pop(drop_feature_index)
    stand_data=stand_scaler.fit_transform(data[data_columns].values)

#下面进行绘制mse指标图片选择特征数量
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
x=np.arange(1,len(cross_score_list)+1)
plt.plot(x,cross_score_list,color='b',linewidth=2,alpha=0.6)
plt.scatter(x,cross_score_list,c='b',alpha=0.6,s=20)
#下面打上标签
for i,value in enumerate(cross_score_list):
    plt.text(x[i],value+0.05,round(value,1),horizontalalignment='center',fontsize=10)
plt.xlabel('Number of features removed',fontsize=13)
plt.ylabel('MSE',fontsize=13)
plt.title('MSE with different number of removal features',fontsize=15)
plt.show()

#进行选择好特征后，可以用柱形图展现选择特征的重要性信息






