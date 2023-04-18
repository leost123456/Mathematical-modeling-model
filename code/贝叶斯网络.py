import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
import copy
from pgmpy.inference import VariableElimination
from collections import Counter
from pgmpy.estimators import K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
import networkx as nx
from sklearn import metrics
import seaborn as sns

#导入数据
data=pd.read_csv('D:\\2022国赛数模练习题\\2022数模国赛\\支撑材料\\data\\处理后的附件1.csv',encoding='gbk')

#下面进行数据预处理
data.drop(columns='文物编号',inplace=True) #丢弃无用的数据列
data.rename(columns={'纹饰':'D','类型':'T','颜色':'C','表面风化':'W'},inplace=True) #更换列名
data['C'].fillna(0,inplace=True) #填补颜色列的缺失值

#下面进行数值化
D_map={'A':0,'B':1,'C':2}
T_map={'高钾':0,'铅钡':1}
C_map={'黑':0,'蓝绿':1,'绿':2,'浅蓝':3,'浅绿':4,'深蓝':5,'深绿':6,'紫':7,0:8}
W_map={'无风化':0,'风化':1}
data['D']=data['D'].map(D_map)
data['T']=data['T'].map(T_map)
data['C']=data['C'].map(C_map)
data['W']=data['W'].map(W_map)

#下面用测试的一个数据集进行自己赋值的一个先验概率（就是把颜色只变成有0/1的）
test_data=copy.deepcopy(data)
test_data['C']=test_data['C'].apply(lambda x:1 if x!=0 else 0)

#1下面进行构建贝叶斯网络（第一种是可以自己来进行构建一个）
model1=BayesianModel([('D','W'),('T','W'),('C','W'),('T','C')]) #这里表示W的父节点为D/T/C,C的父节点有一个T

#2下面进行构建cpd表（每个节点都需要）就是一个条件概率分布表,第一种方式可以根据先验知识进行设置
"""cpd_D=TabularCPD(variable='D', variable_card=3,values=[[0.7], #注意要竖着看
                                                       [0.2],
                                                       [0.1]]) #第一个参数表示节点的名称，第二个参数是节点有几个分类，第三个就是各分类的概率
cpd_T=TabularCPD(variable='T', variable_card=2,values=[[0.6],
                                                       [0.4]])
cpd_C=TabularCPD(variable='C', variable_card=2,values=[[0.8,0.7], #注意要竖着来看，竖着的表示当其父节点（两种选择）分别为0和为1时其取值的各个概率
                                                       [0.2,0.3],],evidence=['T'], evidence_card=[2]) #其中evidence表示其父节点，evidence_card表示父节点有几个分类变量
cpd_W=TabularCPD(variable='W', variable_card=2,values=[[0.1,0.4,0.2,0.4,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.4],
                                                       [0.9,0.6,0.8,0.6,0.5,0.6,0.7,0.8,0.9,0.8,0.7,0.6]],evidence=['D','T','C'], evidence_card=[3,2,2]) #注意其有三个父节点也就是说有3*2*2种类型的条件概率
#下面将概率分布添加到模型中
model1.add_cpds(cpd_D, cpd_T, cpd_C, cpd_W)"""

#第二种方式添加条件概率（利用参数学习的方式）(这时候用原来的C（9个类别的）)
#先统计各个分类变量(节点)的个数，要将其输入到后面参数学习的模型中（注意顺序，要从0开始往后）
"""print(Counter(data['D']))
print(Counter(data['T']))
print(Counter(data['C']))
print(Counter(data['W']))"""
#下面进行参数学习
#pseudo_counts={'D':[[22],[6],[28]],'T':[16,40],'C':[2,15,1,18,3,2,7,4,4],'W':[[22],[34]]}
#model1.fit(data,estimator=BayesianEstimator,prior_type='BDeu') #利用贝叶斯估计器，并用equivalent_sample_size=5无差别客观先验，认定各个概率相等，类似先输入先验的刚开始的正则化操作
#model1.fit(data,estimator=BayesianEstimator,prior_type='dirichlet',pseudo_counts=pseudo_counts) #这个是采用狄利克雷分布，目前这个还用不了
model1.fit(data,estimator=BayesianEstimator,prior_type='K2') #当prior_type= ‘K2’ 意为 ‘dirichlet’ + setting every pseudo_count to 1

#下面可以进行一些检查操作和查看模型参数操作
# 1检查模型的变量
"""print(model1.nodes)
# 2检查模型的边缘依赖关系
print(model1.edges)
# 3检查模型的条件概率分布表
for cpd in model1.get_cpds(): #注意要查看某个节点的话就用model.get_cpds('W').values)
    print(cpd) #用cpd.values可以输出全部的值（矩阵形式）"""

#这样贝叶斯网络模型就构建完毕了，下面就可以进行一些推理和预测
#解决的第一个问题，如果知道其父节点的各个状态（或几个状态），如D是0，T是0，C是1，问W各个值的一个概率（用于分类）
model_infer = VariableElimination(model1)
q1 = model_infer.query(variables=['W'], evidence={'D':0,'T':0,'C':1}) #注意不一定要写出所有的父节点变量（影响因子）
q2 = model_infer.query(variables=['C'],evidence={'T':0}) #注意也可以不给父节点，直接设置子节点，得到所有节点的一个概率分布情况
q3 = model_infer.query(variables=['C']) #直接得到C的概率分布情况，依次类推可以得到所有节点的概率分布情况
print(q1.values,'\nq2为',q2.values)
print('q3',q3.values) #注意可以用q3.values得到其条件概率的一个列表

#2进行最大后验概率问题,比如W为1时，其各个父节点（影响变量）最有可能是什么状态,3最大可能解释问题（MPE）进行因果（概率相关性的判断）,比如我想知道当W等于1，其D为1的一种概率，概率越高说明其实两者之间的相关性越高
q4 =  model_infer.map_query(variables=['D','T','C'],evidence={'W':1})
print(q4)

#下面进行贝叶斯网络的敏感性分析
# 1利用信息熵互信息得到各个自变量与因变量之间的关系程度(研究自变量的变化对因变量的影响)
mutual_list=[] #存储互信息的列表
for column in data.columns[:-1]:
    mutual_list.append(metrics.normalized_mutual_info_score(data['W'], data[column]))

#下面再对互信息进行可视化（绘制一行的热点图）
#先重新构建一个dataframe数据
mutual_dataframe=pd.DataFrame()
for i,column in enumerate(data.columns[:-1]):
    mutual_dataframe[column]=[mutual_list[i]]
print(mutual_dataframe)
mutual_dataframe.index=['W']

#下面开始绘制
sns.set(font_scale=0.7)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax = sns.heatmap(mutual_dataframe.loc[['W'], :], square=True,
                 cbar_kws={'orientation': 'horizontal', "shrink": 0.8},
                 annot=True, annot_kws={"size": 12}, vmax=1.0, cmap='coolwarm')
# 设置标题
ax.set_title('自变量与因变量的互信息',fontsize=13)
# 更改坐标轴标签字体大小
ax.tick_params(labelsize=12)
plt.show()

#2进行场景分析 (可以将设置某一个参数发生的情况下变量的概率分布与某一个参数不发生的情况下变量的概率分布进行对比)
#也可以理解不设置其他的值，就设当此变量出现或者不出现时，因变量各分类变量的概率分布变化百分比
#构建求解百分比的函数（注意模型是在已经构建完成的情况下，里面模型用的是全局变量）
def percent(column_name,target_name,length): #其中column_name是自变量的标签，target_name是因变量的标签,length是自变量中分类变量的个数
    result_perc=[] #存储D中每个分类变量选取与不选取对因变量W的概率分布变化百分比情况（绝对值），形式为[[],[]]
    for i in range(length): #D中有3个分类变量
        f1= model_infer.query(variables=[target_name]).values   #得到原始的分布情况
        f2= model_infer.query(variables=[target_name],evidence={column_name:i}).values #得到当自变量某一个分类变量100%发生时，因变量W概率分布情况
        result_perc.append(np.abs((f2-f1)/f1).tolist()) #得到变化的百分比（绝对值）
    return result_perc #形式为[[],[]]
#下面进行输出结果
D_perc=percent('D','W',3)
T_perc=percent('T','W',2)
C_perc=percent('C','W',2)

#下面进行整体分类的预测(给自变量)
y_pred=model1.predict(data[['D','T','C']],n_jobs=1)
print((y_pred['W']==data['W']).sum()/len(y_pred))  #计算分类的准确度

#下面是利用结构学习的方式得到一个更合理的网络（注意上面的网络是自己设计的，这里进行优化）
#第一种方式是利用评价指标的方法（常见的bdeu，k2，bic），注意指标都是越小越好
k2 = K2Score(data)
bic = BicScore(data)
#构建模型
model1=BayesianModel([('D','W'),('T','W'),('C','W'),('T','C')])
model2=BayesianModel([('D','W'),('T','W'),('C','W')])
#下面进行输出评价指标的值进行比较模型的优度
print(k2.score(model1),k2.score(model2))
print(bic.score(model1),bic.score(model2))

#第二种方式是用查找的方式（智能查找方法比如爬山算法（贪婪算法的一种））
hc = HillClimbSearch(data,state_names=BicScore(data)) #利用爬山算法,评价指标用bic指标
best_model = hc.estimate()
best_edges=best_model.edges() #输出最好的贝叶斯网络结构

#下面用最好的网络进行参数学习并进行预测决策等
best_model=BayesianModel(best_edges) #构建模型

#进行参数学习
best_model.fit(data,estimator=BayesianEstimator,prior_type='K2') #当prior_type= ‘K2’ 意为 ‘dirichlet’ + setting every pseudo_count to 1

#进行决策和预测
best_pred=best_model.predict(data[['D','T','C']],n_jobs=1)
print((best_pred['W']==data['W']).sum()/len(best_pred)) #计算分类的准确度,似乎准确度不高

#下面进行绘制贝叶斯网络结构图(没有带上条件概率的)
pos = nx.spring_layout(best_model) #定义画板
nx.draw_networkx_nodes(best_model, pos, node_size=500,node_color='#2049ff') #绘制各个节点
nx.draw_networkx_edges(best_model, pos, width=2, arrows=True, arrowstyle='->',arrowsize=20,edge_color='#bf0000') #绘制箭头线段
nx.draw_networkx_labels(best_model, pos, font_size=20, font_family='sans-serif') #对节点打上标签
plt.axis('off')
plt.show()