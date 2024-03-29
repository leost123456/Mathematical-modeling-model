import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
from matplotlib import rcParams #导入包
config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

# 准备数据
X = np.random.rand(100, 2)

######可以先利用AIC、BIC或者轮廓系数确定最优组件个数（也可以用于后续的合理性、准确性和敏感性分析）#########
#先进行拟合模型（各个聚类数）
n_series=np.arange(2,10)
models = [GMM(n_components=n,covariance_type='full',random_state=42).fit(X) for n in n_series]
AIC=[m.aic(X) for m in models] #AIC序列
BIC=[m.bic(X) for m in models] #BIC序列
silhouette=[silhouette_score(X,m.predict(X)) for m in models] #轮廓系数

#绘制各个指标的可视化曲线,双坐标轴（注意可以加肘点标记），或者利用单个指标均可
fig, ax1 = plt.subplots(figsize=(8,6)) #设置第一个图
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
plot1=ax1.plot(n_series,AIC,linestyle=':',marker='o',lw=2,c='#007172',label='AIC',markersize=3.5)
plot2=ax1.plot(n_series,BIC,linestyle=':',marker='s',lw=2,c='#f29325',label='BIC',markersize=3.5)
ax1.set_title('聚类数变动的各指标可视化',fontsize=15)
ax1.set_xlabel('Clusters',fontsize=13)
ax1.set_ylabel('AIC or BIC value',fontsize=13)
#ax1.set_ylim(min(y1)-0.1,1.1) #设置显示范围
ax2=ax1.twinx()  #和第一类数据共享x轴（新设置图）
plot3=ax2.plot(n_series,silhouette,linestyle=':',color='#d94f04',marker='*',lw=2,label='Silhouette score',markersize=3.5)
ax2.set_ylabel('Silhouette score',fontsize=13) #设置ax2的y轴参数
#进行打上图例
lines=plot1+plot2+plot3 #目前只能添加线,如果有不同柱形或者其他的，可以手动将图例一个放左边，一个放右边，下面这两行代码就可以去掉（注意对于辅助线来说可以绘制一个空的线对象再进行添加）
ax1.legend(lines,[l.get_label() for l in lines]) #打上图例
#下面进行打上标签（太乱了的话可不加）
for i,data in enumerate(AIC):
    ax1.text(n_series[i],data+0.001,round(data,0),horizontalalignment='center',fontsize=10)
for i,data in enumerate(BIC):
    ax1.text(n_series[i],data+0.02,round(data,0),horizontalalignment='center',fontsize=10)
for i,data in enumerate(silhouette):
    ax2.text(n_series[i],data+0.001,round(data,3),horizontalalignment='center',fontsize=10)
#plt.savefig('',format='svg',bbox_inches='tight')
plt.show()

#######假设聚类数选择3进行绘制聚类结果分布图（概率密度图）############
n=3
gmm=GMM(n_components=3,covariance_type='full',random_state=42)
gmm.fit(X)
print(gmm.covariances_)

#绘制聚类结果图（二维特征情况下）
#给定的位置和协方差画一个椭圆
def draw_ellipse(position,covariance,n_cluster=3,ax=None, **kwargs):
    """
    :param position: 均值
    :param covariance: 方差
    :param n_cluster: 聚类个数
    :param ax: 画板
    :param kwargs: 其余参数例如alpha非透明度
    :return:
    """
    #将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    #画出椭圆
    for nsig in range(1, n_cluster+1):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,angle,**kwargs))

#画图
def plot_gmm(gmm, X,n_cluster=3):
    """
    :param gmm: 训练好的模型
    :param X: 输入特征集
    :param num_cluster: 设定聚类数
    :return:
    """
    labels = gmm.predict(X)
    color_ls=['#007172','#f29325','#d94f04']
    fig,ax = plt.subplots(figsize=(8,6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    #进行绘制散点图
    for i in range(n_cluster): #每个类别
        ax.scatter(X[labels==i][:,0], X[labels==i][:, 1], c=color_ls[i], s=15,label=f'Category{i+1}')
    ax.axis('equal') #画布均衡
    w_factor = 0.2 / gmm.weights_.max() #权重
    #绘制概率密度的椭圆
    for i,(pos, covar, w) in enumerate(zip(gmm.means_,gmm.covariances_,gmm.weights_)):
        draw_ellipse(pos, covar, n_cluster=n_cluster,ax=ax,alpha=w * w_factor,color=color_ls[i])
    plt.legend()
    plt.title('各类别高斯分布拟合结果', fontsize=15)
    plt.xlabel('Feature 1', fontsize=13)
    plt.ylabel('Feature 2', fontsize=13)
    plt.show()

#进行绘制
plot_gmm(gmm, X,n_cluster=3)

##########得到聚类结果##########
labels=gmm.predict(X) #得到各个样本的标签序列
probs = gmm.predict_proba(X) #得到有一个隐含的概率模型。可以通过其得到簇分配结果的概率（将样本分配到各个聚类簇的概率）
print(labels)
print(probs)


