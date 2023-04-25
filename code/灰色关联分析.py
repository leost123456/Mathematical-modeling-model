import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('D:\\2022国赛数模练习题\\data\\附件2.csv',encoding='gbk')
main_data=data['As (μg/g)'] #母序列
data.drop(columns='编号',inplace=True,axis=1)
data.drop(columns='As (μg/g)',inplace=True,axis=1)
data_column=data.columns #列名
#下面进行数据的标准化或者归一化（各种类型的指标处理）
def standardization(data): #data是矩阵,主要功能是进行标准化,输出是经过标准化的矩阵
    data_std=[np.std(data[:,i]) for i in range(data.shape[1])]
    data_mean=[np.mean(data[:,i]) for i in range(data.shape[1])]
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-data_mean[i])/data_std[i]
    return data

# 标准话功能函数(处理序列)
def standardization2(data):  # data是序列,主要功能是进行标准化,输出是经过标准化的序列
    data_std = np.std(data)
    data_mean = np.mean(data)
    for i in range(len(data)):
        data[i] = (data[i] - data_mean) / data_std
    return data

stad_data=pd.DataFrame(standardization(data.values),columns=data_column) #经过处理后的子序列dataframe
main_data=standardization2(main_data) #处理后的母序列

#下面可以进行绘制趋势图(做简单的分析)
plt.figure(figsize=(8,6))
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)  #是否加网格线
x=np.arange(1,len(main_data)+1) #x坐标
plt.plot(x,main_data,ls=':') #绘制母序列图
for i in range(7): #绘制子序列图
    plt.plot(x,stad_data.values[:,i],ls=':')
plt.legend(['As','Cd','Cr','Cu','Hg','Ni','Pb','Zn'])
plt.xlabel('序号',fontsize=13)
plt.ylabel('值',fontsize=13)
plt.title('各个重金属浓度标准后的值',fontsize=15)
plt.show()

#主体函数
def grey_relation_algorithm(main_data,data,p=0.5): #其中main_data是母序列，data是包含所有子序列的dataframe，p是辨别系数，一般取0.5，输入的数据都是要已经过标准化或者正向化处理的数据。
    main_data=np.array(main_data) #转化成向量
    data_matrix=data.values #转化成矩阵
    #下面求解绝对值矩阵（这里假设第一列是母序列）
    for i in range(data_matrix.shape[1]):
        data_matrix[:,i]=np.abs(data_matrix[:,i]-main_data)
    #下面找全局最小点a和最大点b
    a=np.min(data_matrix) #全局最小值
    b=np.max(data_matrix) #全局最大值
    #计算每一个指标的灰色关联度
    result_list=[]
    for i in range(data_matrix.shape[1]):
        result_list.append(np.mean((a+p*b)/(data_matrix[:,i]+p*b)))
    return result_list

result1=grey_relation_algorithm(main_data,stad_data) #输出最终的各指标灰色关联度

#下面进行绘制灰色单行关联度矩阵热点图
#首先要创建一个单行的dataframe
trans_data=pd.DataFrame({'Cd':result1[0],'Cr':result1[1],'Cu':result1[2],'Hg':result1[3],'Ni':result1[4],'Pb':result1[5],'Zn':result1[6]},index=['AS'])
trans_data.index=['As'] #修改行号
#下面开始绘制
sns.set(font_scale=0.7)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 热力图主要参数调整
ax = sns.heatmap(trans_data.loc[['As'], :], square=True,
                 cbar_kws={'orientation': 'horizontal', "shrink": 0.8},
                 annot=True, annot_kws={"size": 12}, vmax=1.0, cmap='coolwarm')
# 设置标题
ax.set_title('各指标的灰色关联度',fontsize=13)
# 更改坐标轴标签字体大小
ax.tick_params(labelsize=12)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=20) #坐标名进行旋转，当坐标名称很长时可使用
plt.show()

#下面是基于灰色关联分析的综合评价法
#首先要对数据进行正向化、归一化或标准化操作
new_data=standardization(data.values) #7列变量

#下面是评价的主体函数
def Gray_comprehensive_evaluation(data,weight=None,p=0.5): #其中data是输入的处理后的矩阵形式的数据，dataframe形式，weight是输入的权重序列，p是分辨系数，默认为0.5
    #下面进行构造虚拟样本序列（求每列的最大值）
    main_vector=np.max(data,axis=0)
    #计算K矩阵（所有样本减去main_vector的绝对值）
    K_matrix=np.abs(data-main_vector)
    #计算全局最小值a和全局最大值b
    a=np.min(K_matrix)
    b=np.max(K_matrix)
    #下面进行计算关联度(每个数都要)
    new_K_matrix=(a+p*b)/(np.abs(K_matrix)+p*b)
    #下面得到每个样本的综合评价分数
    if weight==None: #如果没有给权重的话，就认为所有指标的权重相等
        result=np.sum(new_K_matrix,axis=1)/data.shape[1]
    else: #如果有权重的话
        result=np.sum(new_K_matrix*np.array(weight),axis=1)
    return result #返回各个样本的综合分数，一个序列
#下面进行输出结果
result2=Gray_comprehensive_evaluation(new_data,weight=[0.2,0.1,0.15,0.1,0.1,0.3,0.05],p=0.5)
print(result2)
