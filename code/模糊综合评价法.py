import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import rcParams

#下面进行导入数据
data=pd.read_csv('D:\\2023美赛\\海盗问题\\第二题第三题数据\\total_MS.csv')
data=data[['loss','weapon','num_pirate','P']].values #留下需要的数据，并转化成矩阵的形式
n=data.shape[0] #样本量
m=data.shape[1] #特征数

#先得到各个因素（特征）的权重（可以用熵权法、层次分析法、重要性评分等等）
#下面是用熵权法得到
#首先对各个指标进行正向化操作（这里都是极大型指标）
def Normalized2(data): #data是输入序列,这个是极大型指标的处理方式 输出也是序列
    min_data=min(data)
    max_data=max(data)
    return [(x-min_data)/(max_data-min_data) for x in data]
#下面进行输出
for i in range(m):
    data[:,i]=Normalized2(data[:,i])

def Entropy_weight_method(data): # data是输入的csv文件，输出是权值序列
    yij = data.apply(lambda x: x / x.sum(), axis=0)  # 第i个学生的第j个指标值的比重yij = xij/sum(xij) i=(1,m)
    K = 1 / np.log(len(data))   #常数
    tmp = yij * np.log(yij)
    tmp = np.nan_to_num(tmp)
    ej = -K * (tmp.sum(axis=0))     # 计算第j个指标的信息熵
    wj = (1 - ej) / np.sum(1 - ej)  # 计算第j个指标的权重
    return wj #输出权值序列
#下面得到因素集的权重序列
feature_weights=np.array(Entropy_weight_method(pd.DataFrame(data)))#注意是array的形式

#下面计算原始数据的对各个评价指标的隶属度
#注意这里假设都是1级：0.2 2级 0.4 3级0.6 4级 0.8 5级 1 （也就是1级是0-0.2的）里面取的都是上限，1级用偏小型，2，3，4用中间型，5级用偏大型
#下面用最常用的梯形型的隶属度函数计算（具体见文档）
a=[0.2,0.4,0.6,0.8,1] #评价等级区分（5级，取上界）
result_list=[] #存储每个因素指标的一个评价矩阵R,在这个例子中，其中存储5个因素的隶属评价矩阵
for i in range(m): #每个因素指标
    mid_matrix=np.zeros((n,len(a))) #存储一列中的所有样本的隶属度(对每个指标)
    for j in range(n): #每个样本，一列一列来搞
        for k in range(len(a)): #对每个评价等级
            #注意下面的可以根据需求进行改变每一列的一个隶属度函数计算公式
            if k==0: #第一个评价指标，用偏小型隶属度函数
                if data[j,i] <= a[k]:
                    mid_matrix[j,k]=1
                elif a[k] <= data[j,i] <= a[k+1]:
                    mid_matrix[j,k]=(a[k+1]-data[j,i])/(a[k+1]-a[k])
                elif data[j,i] > a[k+1]:
                    mid_matrix[j,k]=0
            elif k==len(a)-1: #最后一个评价指标,用偏大型隶属度函数
                if data[j,i] <= a[k-1]:
                    mid_matrix[j,k]=0
                elif a[k-1] <= data[j,i] <= a[k]:
                    mid_matrix[j,k]=(data[j,i]-a[k-1])/(a[k]-a[k-1])
                elif data[j,i] > a[k]:
                    mid_matrix[j,k]=1
            else: #中间的评价指标用中间型的隶属度函数
                if a[k-1]<= data[j,i] <= a[k]:
                    mid_matrix[j,k]=(data[j,i]-a[k-1])/(a[k]-a[k-1])
                elif a[k] <= data[j,i] <= a[k+1]:
                    mid_matrix[j,k]=(a[k+1]-data[j,i])/(a[k+1]-a[k])
                else:
                    mid_matrix[j,k]=0
    result_list.append(mid_matrix)

#下面需要从result_list中得到每个样本的一个评价矩阵
evaluate_list=[] #存储每个样本的评价矩阵的列表
for i in range(n): #每个样本
    evaluate_matrix = np.zeros((m, len(a)))  # 创建初始评价矩阵，行数为因素的数量，列数为评价等级指标的数量
    for j in range(m): #每个因素
        evaluate_matrix[j,:]=result_list[j][i,:]
    evaluate_list.append(evaluate_matrix)

#下面进行建立综合评价模型得到每个样本的一个B向量（综合了各个因素后得到的一个评价等级隶属度）
B_list=[] #存储每个 样本的最终评价等级隶属度,形式为[[],[]] 里面的是array形式
for i in range(n): #每个样本
    B_list.append(np.dot(feature_weights,evaluate_list[i]))

#注意可以绘制一些雷达图、柱形图来展现各个最终等级评分的隶属度的均值情况，或者绘制分布图（分类簇状柱形分布）展现各个等级评分隶属度的分布情况
#下面进行绘制各个等级评分的最终隶属度分布情况
class_list=[] #用于隶属度评分等级的标签
for i in range(len(B_list)): #每个样本
    class_list+=['Level 1','Level 2','Level 3','Level 4','Level 5']
#下面构造画图的dataframe数据形式
plot_dataframe=pd.DataFrame({'level':class_list,'value':np.array(B_list).ravel().tolist()})
#下面进行绘制图片
sns.set_theme(style="ticks") #设置一种绘制的主题风格
f, ax = plt.subplots(figsize=(8, 6))
sns.despine(f)
sns.histplot(
    plot_dataframe,
    x="value", hue="level", #注意hue是分类的标签，里面可以是分类的标签（字符和数值型均可）,同时数据标签的前后关系是按照读取数据的先后关系的
    multiple="stack",
    palette="light:m_r",
    edgecolor=".3",
    linewidth=.5,
    log_scale=False, #注意这个是代表进行对数变换，如果原始的值有小于等于0的数就会报错，可以写成False
)
plt.tick_params(size=5,labelsize = 13)  #坐标轴
plt.grid(alpha=0.3)                     #是否加网格线
plt.ylabel('Count',fontsize=13,family='Times New Roman')
plt.xlabel('Membership',fontsize=13,family='Times New Roman')
plt.title('The membership distribution of each evaluation grade of each sample',fontsize=15,family='Times New Roman')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks(np.linspace(0,1,10)) #设置x轴的标签
plt.show()

#下面进行计算总得分
score_list=[5,20,40,70,100] #注意要自己主观对每一等级进行打分
#下面开始计算
F_list=[] #存储每个样本的总得分
for i in range(n):
    F_list.append(np.dot(np.array(score_list),B_list[i].T))
#可以绘制一些散点（加范围选择）或者折线面积图进行展现这个最终的总得分

#注意一些二层、三层指标就是反复调用前面的代码进行操作即可（）

#下面是计算特殊情况的模糊评价分析（适用于直接将各个方案作为各个等级评分的（选方案），然后隶属度的计算就是各个方案的指标）
#创建一个样本数据(假设有4个方案可供选择，有五个指标)
test_data=np.array([[0.2,0.3,0.1,0.3],
                    [0.3,0.4,0.2,0.3],
                    [0.1,0.23,0.4,0.1],
                    [0.12,0.42,0.32,0.14],
                    [0.14,0.16,0.21,0.58]])

#其中data就是输入的数据（矩阵形式）,w_list是各个因素(指标)的权重序列（要和指标的数量对上）
def evalue_func(data,w_list): #其中data就是输入的数据（矩阵形式）,w_list是各个因素(指标)的权重序列
    return np.dot(np.array(w_list),data)
#下面进行输出(选择对应隶属度最高的方案就行了)
test_result=evalue_func(test_data,[0.2,0.3,0.2,0.2,0.1])















