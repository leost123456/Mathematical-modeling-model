from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  #测试的数据集
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
#层次凝聚分类模型
iris=load_iris() #导入数据集 是一个字典类型的数据
X=iris.data[:,2:4] #表示只取特征空间的后两个纬度
y = iris.target     # 将鸢尾花的标签赋值给y
# 随机划分鸢尾花数据集，其中训练集占70%，测试集占30%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#计算各个点之间的距离，搞成一个n*n矩阵
dist = distance_matrix(X, X)
#计算树状图所需参数（两种方式）
Z = hierarchy.linkage(dist, 'complete')
#Z = hierarchy.linkage(dist, 'average')
#绘制树状图确定聚类中心个数
plt.figure(figsize=(6,30))
dendro=hierarchy.dendrogram(Z,leaf_rotation=0,leaf_font_size=5,orientation='right')
plt.show()
#创建层次凝聚聚类模型
agglom = AgglomerativeClustering(n_clusters=3, linkage='average').fit(X) #创建层次凝聚分类模型
output=agglom.labels_ #获取聚类标签
