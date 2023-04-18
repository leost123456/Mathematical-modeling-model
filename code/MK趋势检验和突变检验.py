import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymannkendall as mk #用于mk检验的包
from scipy import stats
from matplotlib import rcParams #导入包
config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#导入数据并进行预处理
data=pd.read_csv('D:\\2023美赛\\海盗问题\\时间序列测试数据.csv',encoding='gbk')
data['Date']=pd.to_datetime(data['Date'])

#下面进行MK趋势检验
result1=mk.original_test(data['num'],alpha=0.05) #导入数据并设置置信度
print(result1) #输出结果
"""
trend就是表示趋势，有'no trend，increasing、decreasing三种类型'
p就是p-value的值   
z就是计算的z值      
Tau是相关性        
s就是计算的s值       
var_s就是s的方差     
slope就是线性回归得到的x前面的参数
"""

#下面进行MK突变检验（检测时间序列数据的突变点（可能也能用于其他类型的数据））
#下面是计算UF和UB的函数
def U_algorithm(data):#其中data是一组数据（一般是时间序列数据）
    n=len(data) #计算序列的长度
    Sk=[0] #Sk序列第一个为0
    Uk=[0] #存储UF或者UB的数据
    s=0 #作为中间数计算累计数
    E=[0] #存储均值的序列
    Var=[0] #存储方差的序列
    #下面计算累计数Sk,还有E和Var
    for i in range(1,n):
        for j in range(i):
            if data[j]<data[i]:
                s+=1
        Sk.append(s)
        E.append((i+1)*(i+2)/4) #注意独立同分布的时候这么计算均值和标准差  Sk[i]的均值
        Var.append((i+1)*i*(2*(i+1)+5)/72) # Sk[i]的方差
        Uk.append((Sk[i]-E[i])/np.sqrt(Var[i])) #计算Uk（UF或者UB）
    Uk=np.array(Uk) #转化成向量形式。
    return Uk

#下面是主体的算法
UFk=U_algorithm(data['num']) #计算出UFk
UBk=U_algorithm(data['num'][::-1]) #计算出UBk，注意原始序列取反再进行计算
UBk=-UBk[::-1] #再进行逆转逆序列
total_U=np.append(UFk,UBk) #整合到一个序列中，用于后续计算最小值和最大值绘制图片做准备

#下面进行输出突变点的位置(由于该数据没有找到突变点，因此我们假设突变点的序号为100)
index=[] #存储突变点序号的序列（可能有多个）
u=UFk-UBk
for i in range(1,len(UFk)):
    if u[i]*u[i-1]<0: #如果两边符号不相等说明取到了交点
        index.append(i) #存储序号

#下面计算95%的置信区间(输出下限值和上限值)
conf_interal=stats.norm.interval(alpha=0.95,loc=0,scale=1) #注意alpha在这里是置信度，也就是0.95，也就是真正的a=0.05,loc是均值，scale是标准误

#下面进行绘制图片(双坐标轴)
fig, ax1 = plt.subplots(figsize=(8,6)) #设置第一个图
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
#下面赋值plot1用于后续的一个标签设置
plot1=ax1.plot(data['Date'],UFk,label='UFk',lw=1.5,color='#fe0000')+\
      ax1.plot(data['Date'],UBk,label='UBk',lw=1.5,color='#bf0000')
#下面进行绘制95%置信区间(注意要打标签只能用plot的方式)先随便绘制个点打标签，再绘制完整的即可
plot3=ax1.plot(data['Date'][1],conf_interal[0], color='black', lw=1.5, linestyle='dashed', label='95% significant level')
ax1.axhline(y=conf_interal[0], color='black', lw=1.5, linestyle='dashed')
ax1.axhline(y=conf_interal[1], color='black', lw=1.5, linestyle='dashed') #包括线和两条置信区间的线
#下面进行绘制突变点位置(假设位置为序号150)
plt.axvline(x=data['Date'][135], color='#52D896',lw=3,linestyle=':')
#下面用箭头并标记突变点的位置
plt.annotate('Mutation position', xy=(data['Date'][135], 9), xytext=(data['Date'][150],9), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
#下面设置坐标轴参数
ax1.set_xlabel('Date',fontsize=13)
ax1.set_ylabel('Statistic',fontsize=13)
ax1.set_title('MK mutation test results',fontsize=15)
ax1.set_ylim(min(total_U)-1,max(total_U)+1) #设置显示范围
#下面进行绘制原始的时间序列数据
ax2 = ax1.twinx()  #和第一类数据共享x轴（新设置图）
plot2=ax2.plot(data['Date'],data['num'],color='#FFDD24',alpha=0.5,lw=1.5,label='Raw data')
ax2.set_ylabel('Count',fontsize=13) #设置ax2的y轴参数
ax2.set_ylim(min(data['num'])-10,max(data['num'])+10)
lines=plot1+plot2+plot3
ax1.legend(lines,[l.get_label() for l in lines]) #打上图例
plt.show()

