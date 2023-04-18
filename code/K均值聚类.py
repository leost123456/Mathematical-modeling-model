import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  #测试的数据集
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns

iris=load_iris() #导入数据集 是一个字典类型的数据
X=iris.data[:,2:4] #表示只取特征空间的后两个纬度(但是这里我取了三个特征纬度)
y = iris.target     # 将鸢尾花的标签赋值给y
# 随机划分鸢尾花数据集，其中训练集占70%，测试集占30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
estimator=KMeans(n_clusters=3) #构造聚类器，将样本聚成3类
estimator.fit(X_train) #开始聚类,注意输入数据是向量的形式
label_pred=estimator.labels_ #获取聚类标签
#print(estimator.cluster_centers_) #获取聚类中心点参数（位置）

#下面将分类标签和特征整合到一个dataframe中便于后续的绘制图形(注意是特征在一列中，数值在一类中，聚类标签也在一列中，一共三列，注意数据的组织方式)
feature_list=[] #存储所有的特征类型
data_list=[]    #存储所有的特征数据
label_list=[]   #也要将横着的标签转化成竖着的
for i in range(X_train.shape[1]): #每个特征
    for j in range(X_train.shape[0]): #每行
        feature_list.append(f'feature{i+1}')
        data_list.append(X_train[j,i])
        label_list.append(label_pred[j])
x_dataframe=pd.DataFrame({'feature':feature_list,'value':data_list,'classify':label_list})
print(x_dataframe)
#下面这个很巧妙，直接将label==0的位置传给了X_train中去，直接读取了出来
x0=X_train[label_pred==0]
x1 = X_train[label_pred == 1]
x2 = X_train[label_pred == 2]

#进行显示聚类结果,绘制k-means结果，将训练集聚类后的结果绘图展示，三种颜色表示三类，红色表示第一类，绿色表示第二类，蓝色表示第三类
#注意特征如果过多的话就绘制不太出来（最多能绘制3维）
plt.figure(figsize=(8,6))
plt.scatter(x0[:,0],x0[:,1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
# 坐标轴属性
plt.xlabel('petal length')  #花瓣属性
plt.ylabel('petal width')   #花瓣宽度
plt.legend(loc=2)
plt.show()

#下面利用训练出来的Kmeans模型（中心点相同）预测测试集中的数据属于哪一类
test_predict=estimator.predict(X_test)
#将各个分类的坐标取出来
predict_0=X_test[test_predict==0]
predict_1=X_test[test_predict==1]
predict_2=X_test[test_predict==2]

#绘制k-means预测结果，将测试集集聚类后的结果绘图展示，三种颜色表示三类，橘色表示第一类，天蓝色表示第二类，蓝绿色表示第三类。
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.scatter(predict_0[:, 0], predict_0[:, 1], c = "tomato", marker='o', label='predict0')
plt.scatter(predict_1[:, 0], predict_1[:, 1], c = "skyblue", marker='*', label='predict1')
plt.scatter(predict_2[:, 0], predict_2[:, 1], c = "greenyellow", marker='+', label='predict2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

#下面可以分特征绘制抖动图（观察聚类效果）
#先把数据转化成dataframe的格式
data_train=pd.DataFrame(X_train,columns=['1','2'])
data_train['result']=label_pred

#下面进行绘制（分特征）
fig=plt.figure(figsize=(8,6))
ax1=fig.add_subplot(121)
sns.swarmplot(x='result',y='1',data=data_train,ax=ax1)

ax2=fig.add_subplot(122)
sns.swarmplot(x='result',y='2',data=data_train,ax=ax2)
plt.show()

#1如果不知道聚类中心数目，为了确定最佳聚类数据可以采用肘方法
clusters = [] #存储SSE
for i in range(2, 12):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
#进行绘制图片（肘图）
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.plot(range(2,12),clusters,linestyle=':',marker='o',lw=3)
plt.title('Searching for Elbow',fontsize=15)
plt.xlabel('Clusters',fontsize=13)
plt.ylabel('Inertia',fontsize=13)
# 打上箭头标签
plt.annotate('Possible Elbow Point', xy=(3, 31.8), xytext=(3, 100), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
plt.show()

#下面采用计算轮廓系数的方法（推荐）
silhouette_factor = [] #存储轮廓系数,
for i in range(2, 12):#注意其不能只分1类，因为要计算离散值（至少2类）
    km = KMeans(n_clusters=i)
    result=km.fit_predict(X)
    silhouette_factor.append(silhouette_score(X,result))

#下面绘制出轮廓系数图，系数在（-1，1）之间越大越好
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.plot(range(2,12),silhouette_factor,linestyle=':',marker='o',lw=3)
plt.title('Searching for Elbow',fontsize=15)
plt.xlabel('Clusters',fontsize=13)
plt.ylabel('Inertia',fontsize=13)
# 打上箭头标签
plt.annotate('Possible Elbow Point', xy=(3, 0.67), xytext=(3, 0.7), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
plt.show()

#下面可以进行绘制分类结果各个特征的分类箱型图进行对比
f,ax=plt.subplots(figsize=(8,6))
sns.boxplot(data=x_dataframe,x='feature',y='value',hue='classify' ,palette="Accent")
#sns.stripplot(data=x_dataframe,x='feature',y='value',hue='classify' ,palette="Accent")#  palette="light:m_r"
#sns.despine(trim=True, left=True)
ax.yaxis.grid(True) #加上y轴的网格（横着的网格）
plt.show()

#下面进行绘制分类均值簇状柱形图（加上误差线）利用seaborn绘制
index=np.arange(x0.shape[1])  #横轴的特征数，这里有两个特征，用于绘制簇状柱形图
feature_index=['feature1','feature2'] #横轴的特征名称
#下面进行计算每一聚类的每一特征的均值
x0_avglist=[np.mean(x0[:,0]),np.mean(x0[:,1])]
x1_avglist=[np.mean(x1[:,0]),np.mean(x1[:,1])]
x2_avglist=[np.mean(x2[:,0]),np.mean(x2[:,1])]
#下面进行计算标准差
x0_stdlist=[np.std(x0[:,0]),np.std(x0[:,1])]
x1_stdlist=[np.std(x1[:,0]),np.std(x1[:,1])]
x2_stdlist=[np.std(x2[:,0]),np.std(x2[:,1])]
#进行绘制
width=0.25
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.bar(index,x0_avglist,width=width,yerr=x0_stdlist,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.4,color='b' ,label = '类别1')
plt.bar(index+width,x1_avglist,width=width,yerr=x1_stdlist,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.4,color='r' ,label = '类别2')
plt.bar(index+2*width,x2_avglist,width=width,yerr=x2_stdlist,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.4,color='g' ,label = '类别3')
#下面进行打上标签
for i,data in enumerate(x0_avglist):
    plt.text(i,data+0.1,round(data,1),horizontalalignment='center',fontsize=13)
#下面进行打上标签
for i,data in enumerate(x1_avglist):
    plt.text(i+width,data+0.1,round(data,1),horizontalalignment='center',fontsize=13)
# 下面进行打上标签
for i, data in enumerate(x2_avglist):
    plt.text(i + width*2, data + 0.1, round(data, 1), horizontalalignment='center',fontsize=13)
plt.xlabel('特征类别',fontsize=13)
plt.ylabel('均值',fontsize=13)
plt.title('聚类类别各特征均值',fontsize=15)
plt.legend()
plt.xticks(index+0.3,feature_index)
plt.show()