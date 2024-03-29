import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib import rcParams #导入包
config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#绘制相关性图函数
def corr_heatmap(data,title): #注意其中data是一个输入的dataframe,title是图片的名称
    corrmat_data=data.corr() #这个是皮尔森相关系数，如果计算斯皮尔曼相关系数，则用method='spearman'
    plt.figure(figsize=(8,6))
    sns.heatmap(corrmat_data,annot=True,cmap='Greens',fmt='.2f') #cmap=Greens、Reds、Blues、Purples 也比较好看
    # 下面是进行设置x轴，y轴刻度字体
    font_prop1 = fm.FontProperties(family="Times New Roman", size=11, )  # 设置一个字体属性的对象（宋体）
    plt.xticks(np.arange(len(data.columns)) + 0.5, data.columns.tolist(), fontproperties=font_prop1, rotation=90) #添加x刻度标签
    plt.yticks(np.arange(len(data.columns)) + 0.5, data.columns.tolist(), fontproperties=font_prop1, rotation=360) #添加y刻度标签
    plt.title(title,fontsize=15,family='SimSun')
    #plt.savefig(title + '.svg', format='svg', bbox_inches='tight')
    plt.show()

def clr_transform(data): #输入是矩阵数据（适用于各种成分数据）
    # 计算每个变量的几何平均值
    geo_means = np.exp(np.mean(np.log(data), axis=0))
    # 计算CLR变换后的数据
    clr_data = np.log(data / geo_means)
    return clr_data

if __name__ == '__main__':
    #导入数据
    boston=load_boston()
    x=boston.data #自变量(总共13个变量)
    y=boston.target #因变量
    x_data = pd.DataFrame(x, columns=boston.feature_names)  # 创建dataframe

    #方法1，利用相关系数矩阵进行观察，相关系数大于0.7或者0.8可以认为有数据共线性
    corr_heatmap(x_data,'皮尔森相关系数矩阵')

    #方法2：利用方差膨胀因子（VIF）进行判断
    vif_ls=[variance_inflation_factor(x,i) for i in range(x.shape[1])] #计算每个自变量的VIF，一一对应
    #下面进行绘制VIF柱形图，加辅助线来直观的观察
    index = np.arange(x.shape[1])  #自变量的数量
    # 进行绘制
    width = 0.25
    plt.figure(figsize=(8, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    # 注意下面可以进行绘制误差线，如果是计算均值那种的簇状柱形图的话(注意类别多的话可以用循环的方式搞)
    plt.bar(index, vif_ls, width=width,color='#bf0000', label='VIF')
    #绘制阈值线和标签
    plt.axhline(y=10, color='black', lw=1.4, linestyle=':')
    plt.axhline(y=100, color='black', lw=1.4, linestyle=':')
    plt.text(x=0.8,y=50,s='中等共线性',color='black',family='SimSun',fontsize=13)
    plt.text(x=0.8, y=105, s='严重共线性', color='black',family='SimSun',fontsize=13)
    # 下面进行打上标签(也可以用循环的方式进行绘制)(颜色就存储在一个列表中)
    for i, data in enumerate(vif_ls):
        plt.text(index[i], data + 0.1, round(data, 1), horizontalalignment='center', fontsize=13)
    plt.xlabel('自变量', fontsize=13,family='SimSun')
    plt.ylabel('VIF', fontsize=13,family='Times New Roman')
    plt.title('各自变量VIF', fontsize=15,family='SimSun')
    plt.legend()
    plt.xticks(index, x_data.columns)
    plt.show()

    # 进行clr变换，创建一个示例数据集，每列代表一个变量，每行代表一个样本
    data = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])

    print(clr_transform(data))






