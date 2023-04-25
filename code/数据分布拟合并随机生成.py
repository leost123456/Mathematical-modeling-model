import pandas as pd
import numpy as np
from fitter import Fitter
import matplotlib.pyplot as plt
from scipy.stats import t
data=pd.read_csv('D:\\2022国赛数模练习题\\data\\转运商数据.csv',encoding='gbk')
data_column=data.columns.tolist()

#先观察下数据的分布
'''plt.figure(figsize=(8,6))
plt.scatter(range(len(data[data_column[0]])),data[data_column[0]])
plt.show()'''

#下面先进行测试
distribution=['norm', 't', 'laplace','gamma', 'rayleigh', 'uniform'] #选择拟合的概率分布函数
f=Fitter(data[data_column[0]],distributions=distribution) #创建拟合对象
f.fit() #进行拟合
f.hist() #绘制组数=bins的标准化直方图
f.plot_pdf(names=None, Nbest=5, lw=2) #绘制分布的概率密度函数
plt.title('6 fitted probability density functions',fontsize=13)
plt.show()
#注意下面返回的是均方误差最小的概率分布函数
print(f.summary()) #返回排序好的分布拟合质量（拟合效果从好到坏）,并绘制数据分布和Nbest分布,注意其是一个dataframe格式的形式的可以从中进行提取数

#绘制AIC和BIC的值（双坐标轴）注意还可以绘制均方误差的图，因为本身是按照那个进行筛选的，一般可以看情况如果一个概率分布有两个都是最的话，就用这两个指标进行操作
x_label=f.summary().index.tolist()
aic=f.summary()['aic'].tolist()  #aic序列
bic=f.summary()['bic'].tolist()  #bic序列
sumsquare_error=f.summary()['sumsquare_error'].tolist() #均方误差序列

fig,ax1=plt.subplots(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
ax1.set_xlabel('Distribution function type',fontsize=13)
ax1.set_ylabel('AIC',fontsize=13)
ax1.plot(x_label,aic,lw=3)
ax1.scatter(x_label,aic,marker='o',s=50)
ax2 = ax1.twinx() #相当于再建立一个坐标轴
ax2.set_ylabel('BIC',fontsize=13)
ax2.plot(x_label,bic,lw=3,color='r')
ax2.scatter(x_label,bic,marker='p',s=50,color='r')

#下面进行打上标签
for i,value in enumerate(aic):
    ax1.text(i,value+4,round(value,2), horizontalalignment='center')
for i,value in enumerate(bic):
    ax2.text(i, value + 0.3, round(value, 2), horizontalalignment='center')

#fig.tight_layout()  # otherwise the right y-label is slightly clipped 这个是自适应调整子图之间的距离，可以不加
plt.title('AIC and BIC comparison of fitted distribution functions',fontsize=15)
plt.show()

#下面运用最优拟合优度的模型产生特征估计值。
#下面进行创建具有最优拟合优度的模型
t_model=t(f.get_best(method='sumsquare_error')['t']) #将t的参数传入，首先要先判断什么是最好的拟合模型再进行导入创建模型
r=t.rvs(200,size=20) #第一个参数是自由度，一般是样本总数减去变量个数
r_zheng=[x for x in r if x>0] #取大于零的数
print(f.get_best(method='sumsquare_error')) #返回最佳拟合分布及其参数
random_num=np.mean(r_zheng) #返回估计值（均值）
print(random_num)

#print(f.fitted_param) #返回拟合分布的参数
#print(f.fitted_pdf) #使用最适合数据分布的分布参数生成的概率密度
#下面的函数的功能：用80种分布进行拟合数据的分布，同时输出最优的前10种参数的AIC,BIC
#def fitted_distribution():
