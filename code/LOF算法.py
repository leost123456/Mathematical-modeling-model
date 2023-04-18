import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from mpl_toolkits.mplot3d import Axes3D
#from __future__ import division
from sklearn.datasets import load_iris  #测试的数据集
from sklearn.neighbors import NearestNeighbors
import networkx as nx

#下面是自构建函数的方式，还可以用sklearn中的函数进行操作
def distance_euclidean(instance1, instance2):
    """Computes the distance between two instances. Instances should be tuples of equal length.
    Returns: Euclidean distance
    Signature: ((attr_1_1, attr_1_2, ...), (attr_2_1, attr_2_2, ...)) -> float"""

    def detect_value_type(attribute): #这个是用于检测输入数据的类型
        """Detects the value type (number or non-number).
        Returns: (value type, value casted as detected type)
        Signature: value -> (str or float type, str or float value)"""
        from numbers import Number
        attribute_type = None
        if isinstance(attribute, Number):
            attribute_type = float
            attribute = float(attribute)
        else:
            attribute_type = str
            attribute = str(attribute)
        return attribute_type, attribute

    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments.")
    # init differences vector
    differences = [0] * len(instance1)
    # compute difference for each attribute and store it to differences vector
    for i, (attr1, attr2) in enumerate(zip(instance1, instance2)):
        type1, attr1 = detect_value_type(attr1)
        type2, attr2 = detect_value_type(attr2)
        # raise error is attributes are not of same data type.
        if type1 != type2:
            raise AttributeError("Instances have different data types.")
        if type1 is float:
            # compute difference for float
            differences[i] = attr1 - attr2
        else:
            # compute difference for string
            if attr1 == attr2:
                differences[i] = 0
            else:
                differences[i] = 1
    # compute RMSE (root mean squared error)
    rmse = (sum(map(lambda x: x ** 2, differences)) / len(differences)) ** 0.5
    return rmse #计算欧氏距离并返回


class LOF:
    """Helper class for performing LOF computations and instances normalization."""
    def __init__(self, instances, normalize=True, distance_function=distance_euclidean):
        self.instances = instances #输入的数据
        self.normalize = normalize #进行标准化
        self.distance_function = distance_function #欧式距离计算
        if normalize:
            self.normalize_instances()

    def compute_instance_attribute_bounds(self):
        min_values = [float("inf")] * len(self.instances[0])  # n.ones(len(self.instances[0])) * n.inf
        max_values = [float("-inf")] * len(self.instances[0])  # n.ones(len(self.instances[0])) * -1 * n.inf
        for instance in self.instances:
            min_values = tuple(map(lambda x, y: min(x, y), min_values, instance))  # n.minimum(min_values, instance)
            max_values = tuple(map(lambda x, y: max(x, y), max_values, instance))  # n.maximum(max_values, instance)
        self.max_attribute_values = max_values
        self.min_attribute_values = min_values

    def normalize_instances(self): #用于标准化数据
        """Normalizes the instances and stores the infromation for rescaling new instances."""
        if not hasattr(self, "max_attribute_values"):
            self.compute_instance_attribute_bounds()
        new_instances = []
        for instance in self.instances:
            new_instances.append(
                self.normalize_instance(instance))  # (instance - min_values) / (max_values - min_values)
        self.instances = new_instances

    def normalize_instance(self, instance): #进行标准化的主体函数
        return tuple(map(lambda value, max, min: (value - min) / (max - min) if max - min > 0 else 0,
                         instance, self.max_attribute_values, self.min_attribute_values))

    def local_outlier_factor(self, min_pts, instance): #计算LOF
        """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
        min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
        Returns: local outlier factor
        Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
        if self.normalize:
            instance = self.normalize_instance(instance)
        return local_outlier_factor(min_pts, instance, self.instances, distance_function=self.distance_function)

#计算第k可达距离
def k_distance(k, instance, instances, distance_function=distance_euclidean):
    # TODO: implement caching
    """Computes the k-distance of instance as defined in paper. It also gatheres the set of k-distance neighbours.
    Returns: (k-distance, k-distance neighbours)
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> (float, ((attr_j_1, ...),(attr_k_1, ...), ...))"""
    distances = {}
    for instance2 in instances:
        distance_value = distance_function(instance, instance2)
        if distance_value in distances:
            distances[distance_value].append(instance2)
        else:
            distances[distance_value] = [instance2]
    distances = sorted(distances.items())
    neighbours = []
    k_sero = 0
    k_dist = None
    for dist in distances:
        k_sero += len(dist[1])
        neighbours.extend(dist[1])
        k_dist = dist[0]
        if k_sero >= k:
            break
    return k_dist, neighbours

#计算可达距离
def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean):
    """The reachability distance of instance1 with respect to instance2.
    Returns: reachability distance
    Signature: (int, (attr_1_1, ...),(attr_2_1, ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])

#计算局部可达密度
def local_reachability_density(min_pts, instance, instances, **kwargs):
    """Local reachability density of instance is the inverse of the average reachability
    distance based on the min_pts-nearest neighbors of instance.
    Returns: local reachability density
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    reachability_distances_array = [0] * len(neighbours)  # n.zeros(len(neighbours))
    for i, neighbour in enumerate(neighbours):
        reachability_distances_array[i] = reachability_distance(min_pts, instance, neighbour, instances, **kwargs)
    sum_reach_dist = sum(reachability_distances_array)
    if sum_reach_dist == 0:
        return float('inf')
    return len(neighbours) / sum_reach_dist

#计算第k离群因子
def local_outlier_factor(min_pts, instance, instances, **kwargs):
    """The (local) outlier factor of instance captures the degree to which we call instance an outlier.
    min_pts is a parameter that is specifying a minimum number of instances to consider for computing LOF value.
    Returns: local outlier factor
    Signature: (int, (attr1, attr2, ...), ((attr_1_1, ...),(attr_2_1, ...), ...)) -> float"""
    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    instance_lrd = local_reachability_density(min_pts, instance, instances, **kwargs)
    lrd_ratios_array = [0] * len(neighbours)
    for i, neighbour in enumerate(neighbours):
        instances_without_instance = set(instances)
        instances_without_instance.discard(neighbour)
        neighbour_lrd = local_reachability_density(min_pts, neighbour, instances_without_instance, **kwargs)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd
    return sum(lrd_ratios_array) / len(neighbours)

#下面的这个是主体的函数计算LOF并输出结果
def outliers(k, instances, **kwargs):
    """Simple procedure to identify outliers in the dataset."""
    instances_value_backup = instances
    inliers = [] #存储正常点的数据
    outliers = [] #存储离群点的数据
    LOF_list= [] #存储所有数据的LOF
    for i, instance in enumerate(instances_value_backup): #进行遍历点的数据
        instances = list(instances_value_backup)
        #instances.remove(instance) #从列表中移除一个个点
        l = LOF(instances, **kwargs) #创建LOF对象
        value = l.local_outlier_factor(k, instance) #计算该点的LOF值
        LOF_list.append(value)
        if value > 1: #如果LOF大于1说明是离群点
            outliers.append({"lof": value, "instance": instance, "index": i}) #输出lof值，原始坐标和序号
        else:
            inliers.append({"lof": value, "instance": instance, "index": i})
    #outliers.sort(key=lambda o: o["lof"], reverse=True) #进行降序
    return inliers,outliers,LOF_list #返回正常点序列、离群点序列和所有数据的LOF数据 前两个参数的形式为[{},{}]，最后的是列表形式

#下面是主体的操作流程
#导入测试数据
data=load_iris().data[:,:2] #只要前面两列数据作为测试

#进行LOF算法
inliers1,outliers1,LOF_list1=outliers(5,data) #返回正常点序列、离群点序列和所有数据的LOF数据 前两个参数的形式为[{},{}]，最后的是列表形式

#下面将正常点和离群点的坐标和LOF值进行读取
inliers_instance=[] #存储正常点的坐标列表[(),()]
inliers_lof=[] #存储正常点的LOF值
outliers_instance=[] #存储离群点的坐标列表[(),()]
outliers_lof=[] #存储离群点的LOF值
for inlier in inliers1:
    inliers_instance.append(inlier['instance'].tolist())
    inliers_lof.append(inlier['lof'])
for outlier in outliers1:
    outliers_instance.append(outlier['instance'].tolist())
    outliers_lof.append(outlier['lof'])

#下面利用线图加点图进行可视化各个位置的LOF值（第一种表现方式（这个并没有显示异常点））
"""x=data[:,0] #x轴数据
y=data[:,1] #y轴数据
#三维点图与线图的结合（可以更加直观的展现数据位置点其他变量值的大小）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,6),facecolor='#ffffff') #创建一个画布窗口
ax=Axes3D(fig) #创建三维坐标系(注意要用Axes中才能改拉伸后面的坐标轴)
#拉伸坐标轴，前3个参数用来调整各坐标轴的缩放比例(目前用不了)
#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1,1 ,1 , 1]))
#下面将背景颜色变透明
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#调整背景景网格粗细
ax.xaxis._axinfo["grid"]['linewidth'] = 0.4
ax.yaxis._axinfo["grid"]['linewidth'] = 0.4
ax.zaxis._axinfo["grid"]['linewidth'] = 0.4
#绘制点图
scatter=ax.scatter(x,y,LOF_list,color='salmon',s=20,alpha=0.8) #注意一定是ax子图中绘制，注意还可以添加cmap参数，但是输输入的c是序列的形式,并且还可以添加颜色条
#绘制竖线
for i in range(len(x)):
    ax.plot([x[i],x[i]],[y[i],y[i]],[min(LOF_list),LOF_list[i]],c='salmon',alpha=0.6,lw=1)
#分别上下旋转和左右旋转，可以自己设置成一个比较好的参数
ax.view_init(20,-32)
#设置z轴的显示范围
ax.set_zlim(min(LOF_list),max(LOF_list)+0.1)
#添加颜色条,注意前面的点图要加cmap,c参数也要改为序列才行
#plt.colorbar(scatter,shrink=0.7)
#设置坐标轴标签
ax.set_xlabel('x轴',fontsize=16)
ax.set_ylabel('y轴',fontsize=16)
ax.set_zlabel('z轴',fontsize=16)
ax.set_title('标题',fontsize=16)
#plt.savefig('测试图',bbox_inches='tight')
plt.show()"""

#下面绘制可以表现出离群点的散点图（气泡图）
def Normalized2(data): #data是输入序列,这个是极大型指标的处理方式 输出也是序列
    min_data=min(data)
    max_data=max(data)
    return [(x-min_data)/(max_data-min_data) for x in data]
#下面进行图片的一些基础设置
"""plt.figure(figsize=(8,6))     
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
#下面进行构建合适的点大小     
len_outliers=len(outliers_instance) #统计离群点的个数
total_lof=inliers_lof+outliers_lof #将正常点和离群点放入一个列表中
size=np.array(Normalized2(total_lof))*400
#下面进行分别绘制正常点和离群点
plt.scatter([x[0] for x in inliers_instance],[x[1] for x in inliers_instance],s=size[:-len_outliers],alpha=0.9,c='#F21855',label='正常点')
plt.scatter([x[0] for x in outliers_instance],[x[1] for x in outliers_instance],s=size[-len_outliers:],alpha=1,c='k',label='离群点')
plt.legend()
plt.xlabel('x轴',fontsize=13)
plt.ylabel('y轴',fontsize=13)
plt.title('标题',fontsize=15)
#plt.savefig()
plt.show()"""

#下面进行绘制近邻网络图
"""X = data #数据（矩阵类型（n*2））
# 生成K近邻网络
nbrs = NearestNeighbors(n_neighbors=6).fit(X) #进行训练，表示6个最近邻的（注意这个是包括自己的），因此当前面LOF为5时，这里是6
distances, indices = nbrs.kneighbors(X) #distances表示每个点到最近的几个点的距离是一个（n,k_neighbors）的一个矩阵形式,indices表示每个点离自己最近的k个点的序号（包括自己）
#下面进行绘制邻接矩阵
plt.figure(figsize=(8,6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
# 绘制点图
plt.scatter(X[:, 0], X[:, 1], s=30, c='#F21855', alpha=1,marker='o',label='正常点')
#标记下异常点
plt.scatter([x[0] for x in outliers_instance],[x[1] for x in outliers_instance],s=55,c='k',alpha=0.8,label='异常点')
# 绘制边
for i in range(len(X)):
    for j in indices[i,1:]:
        plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], c='gray', alpha=1)
plt.legend()
plt.xlabel('x轴',fontsize=13)
plt.ylabel('y轴',fontsize=13)
plt.title('标题',fontsize=15)
plt.show()"""

#下面对k参数进行敏感性分析（绘制6张散点图）
k_list=[5,10,15,20,25,30]
fig,ax=plt.subplots(2,3,figsize=(16,8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
for i,k in enumerate(k_list):
    inliers2,outliers2,LOF_list1=outliers(k,data) #返回正常点序列、离群点序列和所有数据的LOF数据 前两个参数的形式为[{},{}]，最后的是列表形式
    #下面将正常点和离群点的坐标和LOF值进行读取
    inliers_instance1=[] #存储正常点的坐标列表[(),()]
    inliers_lof1=[] #存储正常点的LOF值
    outliers_instance1=[] #存储离群点的坐标列表[(),()]
    outliers_lof1=[] #存储离群点的LOF值
    for inlier in inliers2:
        inliers_instance1.append(inlier['instance'].tolist())
        inliers_lof1.append(inlier['lof'])
    for outlier in outliers2:
        outliers_instance1.append(outlier['instance'].tolist())
        outliers_lof1.append(outlier['lof'])
    #下面进行绘制图形（还可以写上有多少个异常值）
    ax[i//3][i%3].tick_params(size=5, labelsize=13)  # 坐标轴
    ax[i//3][i%3].grid(alpha=0.3)  # 是否加网格线
    ax[i//3][i%3].scatter([x[0] for x in inliers_instance1],[x[1] for x in inliers_instance1],s=35,c='#F21855',alpha=0.8,label='正常点')
    ax[i//3][i%3].scatter([x[0] for x in outliers_instance1],[x[1] for x in outliers_instance1],s=70,c='k',alpha=1,label='异常点')
    ax[i//3][i%3].legend()
    ax[i//3][i%3].set_xlabel('x轴', fontsize=13)
    ax[i//3][i%3].set_ylabel('y轴', fontsize=13)
    ax[i//3][i%3].set_title(f'k={k}(异常值数目为{len(outliers_instance1)})',fontsize=15)
fig.suptitle('参数k的敏感性分析',fontsize=15)
plt.tight_layout()
plt.show()






