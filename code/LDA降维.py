from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams #导入包

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

# 生成示例数据集(1000行，7列数据,最后一列是类别，四类)(已经标准化过后的数据)
# n_informative表示与类别相关的特征列数量，n_redundant是冗余特征的数量，冗余特征是信息特征的随机线性组合（不会增加模型中的新信息）,n_clusters_per_class表示每类别中的族数（加大分类难度）
X, y = make_classification(n_samples=1000, n_features=6, n_classes=4,n_informative=3, n_redundant=1, n_clusters_per_class=2, random_state=42)

# 将数据和类别标签合并到DataFrame中（1000*7）
data = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=[f'Feature_{i+1}' for i in range(6)] + ['Class'])

# 使用LDA进行降维
X = data.drop('Class', axis=1)  # 特征数据
y = data['Class']  # 类别数据

# 初始化LDA并将数据降维到3列（注意最多只能降维到类别数-1维）
lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit_transform(X, y) #数据为矩阵形式

# 将降维后的数据转换为DataFrame
data_lda = pd.DataFrame(X_lda, columns=['Component_1', 'Component_2', 'Component_3'])

# 将降维后的数据和类别标签合并到一起
data_final = pd.concat([data_lda, data['Class']], axis=1)

#绘制三维点图进行展示降维效果（数据分布情况）（不同标签不同颜色）
fig = plt.figure(figsize=(8,6)) #创建一个画布窗口
ax=plt.axes(projection='3d') #注意现在要用这种3d绘图创建方式（直接设置三维坐标轴）
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#调整背景景网格粗细
ax.xaxis._axinfo["grid"]['linewidth'] = 0.4
ax.yaxis._axinfo["grid"]['linewidth'] = 0.4
ax.zaxis._axinfo["grid"]['linewidth'] = 0.4

#开始绘制
color_ls=['#fe0000','#bf0000','#2049ff','#febf00']
for i in range(4): #每个类别
    #取出各类别下的三个特征
    x=data_final[data_final['Class']==i]['Component_1'].tolist()
    y=data_final[data_final['Class']==i]['Component_2'].tolist()
    z = data_final[data_final['Class'] == i]['Component_3'].tolist()
    ax.scatter(x,y,z,color=color_ls[i],s=20,label=f'Class{i+1}') #绘制

#分别上下旋转和左右旋转，可以自己设置成一个比较好的参数
ax.view_init(65,-32)

#设置坐标轴
plt.legend()
ax.set_xlabel('特征1',fontsize=15,family='SimSun')
ax.set_ylabel('特征2',fontsize=15,family='SimSun')
ax.set_zlabel('特征3',fontsize=15,family='SimSun')
ax.set_title('LDA降维后特征可视化',fontsize=16,family='SimSun')
#plt.savefig('测试图',format='svg',bbox_inches='tight')
plt.show()

