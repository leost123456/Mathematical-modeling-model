import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  #测试的数据集
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

iris=load_iris() #导入数据集 是一个字典类型的数据
X=iris.data[:,2:4] #表示只取特征空间的后两个纬度(但是这里我取了三个特征纬度)
y = iris.target     # 将鸢尾花的标签赋值给y
#首先进行数据的标准化
def standardization(data): #data是矩阵,主要功能是进行标准化,输出是经过标准化的矩阵
    data_std=[np.std(data[:,i]) for i in range(data.shape[1])]
    data_mean=[np.mean(data[:,i]) for i in range(data.shape[1])]
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-data_mean[i])/data_std[i]
    return data
scale_X_dataframe=pd.DataFrame(standardization(X),columns=['feature1','feature2']) #对数据进行标准化
#注意如果特征维数过高可以运用降维算法对数据进行降维（但是要确保降维后的数据具有解释意义）

#下面进行确定两个核心参数（最大半径epsilon和一个簇中最少点数minPts）
#确定epsilon参数，第一种方式（利用肘型图）(注意minPts可以根据经验法则为特征维数的两倍)
nn=NearestNeighbors(n_neighbors=5).fit(scale_X_dataframe) #创建对象，设置K参数，这边是传入被算距离的数据(注意有还可以做一下这个k参数的敏感性分析)
distance,index=nn.kneighbors(scale_X_dataframe) #这里是计算其中的每一个点与原来训练的数据之间每个点的距离，index是与数据最近的训练数据的索引值
#下面进行计算周围K个点的平均距离
mean_distance=np.mean(distance,axis=1)#横向求
#下面进行排序
mean_distance_sort=sorted(mean_distance) #从小到大
#下面进行绘制肘性图进行观察（选择突变的情况就是突然增大的时候）
plt.figure(figsize=(8,6))
#plt.rc('font',family='Times New Roman')  #更改画图字体为Times New Roman
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·adwdsdsdsdfgh
plt.plot(np.arange(len(mean_distance_sort)),mean_distance_sort,alpha=0.8,lw=3)
#plt.scatter(np.arange(len(mean_distance_sort)),mean_distance_sort,alpha=0.5,color='b',s=10)
plt.annotate('Mutation point', xy=(140, 0.1668), xytext=(110, 0.2),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2)) #在图中进行标注
plt.xlabel('index',fontsize=13)
plt.ylabel('Average Distance',fontsize=13)
plt.title('The Average Distance From k Points',fontsize=15)
plt.show() #可以观察到epsilon差不多可以取0.17左右

#第二种方式（利用迭代的思想寻找最优的epsilon和minPts）评价指标是聚类后的轮廓系数
iteration_result=[] #存储最后的结果（里面可以再进行嵌套一层列表存储当前的epsilon、minPts和轮廓系数）
epsilon_iter=np.linspace(0.05,0.3,10) #最大半径的迭代数据
minPts_iter=np.arange(2,9) #最小点数的迭代数据
for epsilon in epsilon_iter:
    for minPts in minPts_iter:
        DBSCAN_cluster=DBSCAN(eps=epsilon,min_samples=minPts).fit(scale_X_dataframe) #创建DBSCAN模型同时导入参数进行聚类
        cluster_num=len(np.unique(DBSCAN_cluster.labels_)) #计算聚类个数(注意其中是包含离群点类别的(如果全部都有类别的话也不一定有离群点，反正要注意观察))
        sil_score=silhouette_score(scale_X_dataframe,DBSCAN_cluster.labels_) #计算轮廓系数
        iteration_result.append([epsilon,minPts,cluster_num,sil_score]) #存储数据，分别是最大半径、最小点数、聚类数、轮廓系数

#下面可以进行可视化操作，（将寻找最优解的过程进行可视化（可以绘制三维图片））
epsilon_list=[x[0] for x in iteration_result] #存储最大半径列表
minPts_list=[x[1] for x in iteration_result] #存储最少数量点列表
sil_list=[x[3] for x in iteration_result] #存储轮廓系数列表

#下面开始进行绘制
fig = plt.figure(figsize=(7,5),facecolor='#ffffff') #创建一个画布窗口
ax=Axes3D(fig) #创建三维坐标系(注意要用Axes中才能改拉伸后面的坐标轴)
#下面将背景颜色变透明
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#调整背景景网格粗细
ax.xaxis._axinfo["grid"]['linewidth'] = 0.4
ax.yaxis._axinfo["grid"]['linewidth'] = 0.4
ax.zaxis._axinfo["grid"]['linewidth'] = 0.4
#绘制点图
ax.scatter(epsilon_list,minPts_list,sil_list,color='salmon',s=20,alpha=0.8,) #注意一定是ax子图中绘制
#绘制竖线
for i in range(len(epsilon_list)):
    ax.plot([epsilon_list[i],epsilon_list[i]],[minPts_list[i],minPts_list[i]],[min(sil_list),sil_list[i]],c='salmon',alpha=0.6,lw=2)
#分别上下旋转和左右旋转，可以自己设置成一个比较好的参数
ax.view_init(20, -32)
#设置z轴的显示范围(注意显示范围)
ax.set_zlim(min(sil_list),max(sil_list))
#设置坐标轴标签
ax.set_xlabel('Epsilon',fontsize=13)
ax.set_ylabel('MinPts',fontsize=13)
ax.set_zlabel('silhouette coefficient',fontsize=13)
ax.set_title('Silhouette Coefficient For Different Epsilon And MinPts',fontsize=15)
plt.show()

#下面进行输出最优的参数(根据轮廓系数进行降序排序)
iteration_result_sort=sorted(iteration_result,reverse=True,key=lambda x: x[3])
best_parameter=iteration_result_sort[2] #存储最优的参数
print(iteration_result_sort)
#根据聚类数轮廓数的最佳，因此我们选择[0.3, 4, 2, 0.743371950414545] 第一个参数是最大半径，第二个参数是最少数量，第三个是聚类数，第四个是轮廓系数

#下面进行最终聚类
DBSCAN_cluster=DBSCAN(eps=best_parameter[0], min_samples=best_parameter[1]).fit(scale_X_dataframe)
result_data=pd.DataFrame(X,columns=['feature1','feature2'])
result_data['label']=DBSCAN_cluster.labels_

#下面进行模型的合理性分析（绘制数据分布图） （针对这种只有两个特征可以直接绘制二维图或者绘制分类矩阵图，或者选取特征进行绘制）
#1分类矩阵图
sns.pairplot(result_data, hue='label', aspect=1.5)
plt.show()
#2特征分类均值簇状柱形图（利用seaborn）具体可参考kmeans-聚类中的绘制过程
#3特征分类箱型图（具体可参考k-means中的绘制过程）

#下面进行参数敏感性分析（两个核心参数参数）保持一个参数最佳，另外一个参数变化，注意还可以做一下k的最近距离点的平均值的敏感性分析
#先进行epsilon参数的敏感性分析
epsilon_list1=np.linspace(0.05,0.3,10)
sil_list1=[] #存储轮廓系数
cluster_num_list1=[] #存储聚类数目
for epsilon in epsilon_list1:
    DBSCAN_cluster = DBSCAN(eps=epsilon, min_samples=best_parameter[1]).fit(scale_X_dataframe)  # 创建DBSCAN模型同时导入参数进行聚类
    cluster_num_list1.append(len(np.unique(DBSCAN_cluster.labels_)))  # 计算并存储聚类个数(注意其中是包含离群点类别的(如果全部都有类别的话也不一定有离群点，反正要注意观察))
    sil_list1.append(silhouette_score(scale_X_dataframe, DBSCAN_cluster.labels_))  # 计算并存储轮廓系数
#下面进行绘制结果图(双坐标轴，一个表示轮廓系数，另外一个表示聚类数)
#config = {"font.family":'Times New Roman'}  # 设置字体类型
#rcParams.update(config)
fig, ax1 = plt.subplots(figsize=(8,6)) #设置第一个图
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
#plt.rc('font',family='Times New Roman')  #更改画图字体为Times New Roman
plot1=ax1.plot(epsilon_list1,sil_list1,marker='o',alpha=0.5,color='b',lw=2,label='silhouette coefficient Curve')
ax1.set_xlabel('epsilon',fontsize=13) #设置x轴参数
ax1.set_ylabel('silhouette coefficient',fontsize=13) #设置y轴的名称
ax1.set_title('Sensitivity Analysis Of Epsilon',fontsize=15)
ax1.tick_params(axis='y', labelcolor='b',labelsize=13) #设置第一个类型数据y轴的颜色参数
ax1.set_ylim(-0.3,0.9) #调整y轴的显示坐标
ax2 = ax1.twinx()  #和第一类数据共享x轴（新设置图）
plot2=ax2.plot(epsilon_list1,cluster_num_list1,color='r',alpha=0.5,marker='*',lw=2,label='Cluster Number Curve')
ax2.set_ylabel('Cluster Number',fontsize=13) #设置ax2的y轴参数
ax2.tick_params(axis='y', labelcolor='r',labelsize=13) #设置ax2的y轴颜色
ax2.set_ylim(1.8,11)
lines=plot1+plot2
ax1.legend(lines,[l.get_label() for l in lines]) #这边进行打上图例
#下面进行打上标签
for i,data in enumerate(sil_list1):
    ax1.text(epsilon_list1[i],data+0.001,round(data,3),horizontalalignment='center',fontsize=10)
for i,data in enumerate(cluster_num_list1):
    ax2.text(epsilon_list1[i],data+0.1,round(data,0),horizontalalignment='center',fontsize=10)
plt.show()

#接下来是minPts的敏感性分析
minPts_list1=np.arange(2,16)
sil_list2=[] #存储轮廓系数
cluster_num_list2=[] #存储聚类数目
for minPts in minPts_list1:
    DBSCAN_cluster = DBSCAN(eps=best_parameter[0], min_samples=minPts).fit(scale_X_dataframe)  # 创建DBSCAN模型同时导入参数进行聚类
    cluster_num_list2.append(len(np.unique(DBSCAN_cluster.labels_)))  # 计算并存储聚类个数(注意其中是包含离群点类别的(如果全部都有类别的话也不一定有离群点，反正要注意观察))
    sil_list2.append(silhouette_score(scale_X_dataframe, DBSCAN_cluster.labels_))  # 计算并存储轮廓系数
#下面进行绘制结果图(双坐标轴，一个表示轮廓系数，另外一个表示聚类数)
fig, ax1 = plt.subplots(figsize=(8,6)) #设置第一个图
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
#plt.rc('font',family='Times New Roman')  #更改画图字体为Times New Roman
plot1=ax1.plot(minPts_list1,sil_list2,marker='o',alpha=0.5,color='b',lw=2,label='silhouette coefficient Curve')
ax1.set_xlabel('MinPts',fontsize=13) #设置x轴参数
ax1.set_ylabel('silhouette coefficient',fontsize=13) #设置y轴的名称
ax1.set_title('Sensitivity Analysis Of MinPts',fontsize=15)
ax1.tick_params(axis='y', labelcolor='b',labelsize=13) #设置第一个类型数据y轴的颜色参数
#ax1.set_ylim(0.45,0.8)
ax2 = ax1.twinx()  #和第一类数据共享x轴（新设置图）
plot2=ax2.plot(minPts_list1,cluster_num_list2,color='r',alpha=0.5,marker='*',lw=2,label='Cluster Number Curve')
ax2.set_ylabel('Cluster Number',fontsize=13) #设置ax2的y轴参数
ax2.tick_params(axis='y', labelcolor='r',labelsize=13) #设置ax2的y轴颜色
#ax2.set_ylim(1.8,5)
lines=plot1+plot2
ax1.legend(lines,[l.get_label() for l in lines]) #这边进行打上标签
#下面进行打上标签
for i,data in enumerate(sil_list2):
    ax1.text(minPts_list1[i],data+0.001,round(data,3),horizontalalignment='center',fontsize=10)
for i,data in enumerate(cluster_num_list2):
    ax2.text(minPts_list1[i],data+0.02,round(data,0),horizontalalignment='center',fontsize=10)
plt.show()

#下面还可以进行离的最近的个数参数k的敏感性分析（绘制肘形图）（如果前面确定参数是利用肘形图的话可以做一下这个敏感性分析，如果是用迭代出最优的话，就不需要了）

















