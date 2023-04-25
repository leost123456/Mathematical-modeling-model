import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from fitter import Fitter
from matplotlib import cm

#进行前向回归分析的函数(用于多元线性回归)
def forward_selected(data, response): #其中data是包含需要回归的数据（dataframe格式），response是因变量的名称
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

##################################
#下面进行一元多项式拟合
##################################

#一个一个进行拟合函数得到三个多项式函数
data_customer_rate=pd.read_csv('D:\\2022国赛数模练习题\\贷款年利率与客户流失率关系.csv',encoding='gbk')
#创建待拟合的数据
x=data_customer_rate['贷款年利率']
y1=data_customer_rate['信誉评级A']
#创建多项式拟合函数得到方程
z1=np.polyfit(x,y1,4)#第三个参数表示多项式最高次幂
#将多项式带入方程，得到多项式函数
p1=np.poly1d(z1)
x1=np.linspace(x.min(),x.max(),100)#x给定数据太少，方程曲线不光滑，多取x值得到光滑曲线
pp1=p1(x1) #将x1带入多项式得到多项式曲线（得到对应的函数值）

#下面还可绘制回归曲线
pp2=pp1
pp3=pp2
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(13,8))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.plot(data_customer_rate['贷款年利率'],data_customer_rate['信誉评级A'],label='信誉评级A',linewidth=2,alpha=0.8,linestyle=':')
plt.plot(data_customer_rate['贷款年利率'],data_customer_rate['信誉评级B'],label='信誉评级B',linewidth=2,alpha=0.8,linestyle=':')
plt.plot(data_customer_rate['贷款年利率'],data_customer_rate['信誉评级C'],label='信誉评级C',linewidth=2,alpha=0.8,linestyle=':')
plt.plot(x1,pp1,linewidth=2,alpha=0.8,label='信誉评级A的多项式曲线')
plt.plot(x1,pp2,linewidth=2,alpha=0.8,label='信誉评级B的多项式曲线')
plt.plot(x1,pp3,linewidth=2,alpha=0.8,label='信誉评级C的多项式曲线')
plt.xlabel('贷款年利率',fontsize=13)
plt.ylabel('客户流失率',fontsize=13)
plt.title('贷款年利率与客户流失率关系图',fontsize=15)
plt.legend()
plt.show()
#注意后续可以求R方、均方误差等指标进行模型的检验

#################################################
#下面进行自定义函数回归分析（一元的）
#################################################

#先自定义一个函数例如：其中有a,b，c三个参数需要回归得出
def func(x, a, b, c,d):
  return a*x**3+b*x**2+c*x**1+d

#接下来直接回归出参数即可（用curve_fit）其中xdata是输入x值，ydata是输入y值，popt是输出的参数
popt, pcov = curve_fit(func, x, y1,maxfev=5000) #maxfev是寻找迭代的次数，如果迭代了maxfev次还未找到最优参数就会退出
print(popt)
print(pcov)
#下面将参数带入就可得到拟合的方程，再将x值带入
plt.figure(figsize=(8,6))
plt.plot(x, func(x, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f，d=%5.3f' % tuple(popt))
plt.plot(x,y1)
plt.legend()
plt.show()

#下面进行模型合理性分析，用R方（两种方式，第一种是Pearson相关系数的平方，第二种是）
def computeCorrelation(x, y): #其中x,y均为序列，x是预测值，y是真实值,这里是计算Pearson相关系数，最后需要平方注意
    xBar = np.mean(x) #求预测值均值
    yBar = np.mean(y) #求真实值均值
    covXY = 0
    varX = 0          #计算x的方差和
    varY = 0          #计算y的方差和
    for i in range(0, len(x)):
        diffxx = x[i] - xBar  #预测值减预测值均值
        diffyy = y[i] - yBar  #真实值减真实值均值
        covXY += (diffxx * diffyy)
        varX += diffxx ** 2
        varY += diffyy ** 2
    return covXY/np.sqrt(varX*varY)

#第二种计算R方的方式
def R_Squre2(y_pred,y_real): #其中y_pred是预测值，y_real是真实值,两者都是徐磊
    y_real_mean = np.mean(y_real)
    y_pred_var=0 #计算预测值减真实值的平方和
    y_real_var=0
    for i in range(len(y_pred)):
        y_pred_var+=(y_pred[i]-y_real_mean)**2
        y_real_var+=(y_real[i]-y_real_mean)**2
    return y_pred_var/y_real_var

#下面进行计算均方误差（MSE）
def mean_squared_error(y_pred,y_real): #其中y_pred是预测值，y_real是真实值
    result_list=[]
    for i in range(len(y_pred)):
        result_list.append((y_pred-y_real)**2)
    return np.sum(result_list)/len(result_list)

#AIC准则（越小越好）
def cal_AIC(n,mse,num_params):#其中n是观测数量，mse是均方误差，num_params是模型参数个数
    aic=n*np.log(mse)+2*num_params
    return aic

#BIC准则（越小越好，表示模型性能越好和复杂度越低）
def cal_BIC(n,mse,num_params):#其中n是观测数量，mse是均方误差，num_params是模型参数个数
    bic=n*np.log(mse)+num_params*np.log(n)
    return bic

y_pred=func(x,*popt) #计算拟合函数的预测值
R2_1=computeCorrelation(y_pred,y1)**2 #一定要平方注意计算R方
R2_2=R_Squre2(y_pred,y1) #j计算第二种R方
MSE=mean_squared_error(y_pred,y1) #计算均方误差
aic=cal_AIC(len(y_pred),MSE,4) #计算AIC
bic=cal_BIC(len(y_pred),MSE,4) #计算BIC
print('第一种R方',R2_1) #越小越好
print('第二种R方',R2_2) #越小越好
print('均方误差',MSE)
print('AIC',aic)
print('BIC',bic)

##############################################################
#下面进行多元自定义函数回归（如果是三维点的话可以自己自定义多元多项式回归）
##############################################################

#首先导入数据
data=pd.DataFrame(pd.read_excel('D:\\国赛数模优秀论文\\2017\\Problems\\B\\附件一：已结束项目任务数据.xls'))
#选择有用的数据
data=data[['任务gps 纬度','任务gps经度','任务标价']]

#下面进行二元多项式回归（x2,x,xy,y,y2）设置函数，总共六个参数
def func(data,a,b,c,d,e,f): #注意data是一个dataframe格式的数据，取第一列和第二列为自变量
    return a*data.iloc[:,0]*data.iloc[:,0] +b*data.iloc[:,0] +c*data.iloc[:,1]*data.iloc[:,1]+ \
           d*data.iloc[:, 1]+e*data.iloc[:, 0]*data.iloc[:, 1] +f

#下面进行函数拟合操作(回归出参数)
params, pcov = curve_fit(func, data.iloc[:,:2], data.iloc[:,2])

#下面进行绘制三维拟合曲面
x=np.linspace(np.min(data.iloc[:,0]),np.max(data.iloc[:,0]),200)
y=np.linspace(np.min(data.iloc[:,1]),np.max(data.iloc[:,1]),200)
X,Y=np.meshgrid(x,y)
#下面是关键点，直接乘得到Z注意这个Z是二维形式的
Z=params[0]*X*X+params[1]*X+params[2]*Y*Y+params[3]*Y+params[4]*X*Y+params[5]
#下面得到一维形式的z
z=params[0]*data.iloc[:,0]*data.iloc[:,0]+params[1]*data.iloc[:,0] +params[2]*data.iloc[:,0]*data.iloc[:,1]+\
  params[3]*data.iloc[:,1]+params[4]*data.iloc[:,1]*data.iloc[:,1]+params[5]

#下面进行绘制三维图形
#plt.style.use('') #目前可改变3D图像样式的有ggplot、seaborn、Solarize_Light2、grayscale
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,6),facecolor='#ffffff')
ax = fig.gca(projection='3d')
#绘制表面图
surf = ax.plot_wireframe(X, Y, Z,rstride=4, cstride=4,color='black',alpha=0.3,
                       linewidth=0.05, antialiased=False)
#ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0.1, antialiased=False)

#下面是进行三维的三角型拟合得出最终的三维图形
"""surf=ax.plot_trisurf(data.iloc[:,0],data.iloc[:,1],z,
                cmap=cm.coolwarm, edgecolor='none')"""
#绘制等高线
#cset1 = ax.contourf(X, Y, Z, zdir='z', cmap=cm.coolwarm)
#cset2= ax.contourf(X, Y, Z, zdir='x' ,cmap=cm.coolwarm)
#cset3 = ax.contourf(X, Y, Z, zdir='y', cmap=cm.coolwarm)

#下面将散点绘制到图片中去
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],marker='o',c=data.iloc[:,2],cmap=cm.coolwarm)
# Add a color bar which maps values to colors.(这里加上颜色条，同时设置其参数)
#fig.colorbar(surf, shrink=0.5, aspect=5,ticks=np.linspace(81,100,5))
#打上x、y、z轴的标签和标题
ax.set_xlabel('X Label',fontsize=13)
ax.set_ylabel('Y Label',fontsize=13)
ax.set_zlabel('Z Label',fontsize=13)
ax.set_title('3D Fitted Surface Plot',fontsize=15)
plt.show()


