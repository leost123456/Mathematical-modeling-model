import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,DotProduct,WhiteKernel
from matplotlib import cm

#导入数据
data=pd.read_csv('D:\\2022国赛数模练习题\\贷款年利率与客户流失率关系.csv',encoding='gbk')
x=data.iloc[:,0].values.reshape(-1,1) #注意要转化成二维的，相当于列向量
y=data.iloc[:,1].values.reshape(-1,1)

#初始化核函数(其中ContantKernel相当于对均值会进行改变的核函数) 其中constant_value是设置的初始值，后面的bounds是参数调优的范围
kernel=ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) *\
       RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4)) #这个是径向基内核，第一个参数是缩放系数，更多的核函数可以看文档说明
#定义高斯过程回归模型
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#＃使用参数的最大似然估计来拟合数据
gpr.fit(x,y)
#进行预测数据
x_test=np.linspace(min(x),max(x),1000) #注意这里是二维的数据（-1，1），因为受原始的x的影响，进行取1000个数，后续画函数图像更加平滑
mu,cov=gpr.predict(x_test,return_cov=True)
y_test=mu.ravel() #将均值向量展平
uncertainty = 1.96 * np.sqrt(np.diag(cov)) #计算95%置信区间，用于后续绘图

#下面进行绘制拟合曲线
# plotting
x_test=x_test.ravel()    #注意这里将数据进行展平
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
plt.fill_between(x_test, y_test + uncertainty, y_test - uncertainty,alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.plot(x_test, y_test, label="Gaussian Process Fitting Curve")
plt.scatter(x, y, label="raw data points", c="red", marker="x")
plt.legend()
plt.show()

#得出回归函数的R方
print(gpr.score(x,y))

#下面进行高维数据的拟合（这里采用三维数据）
#首先导入数据
data=pd.DataFrame(pd.read_excel('D:\\国赛数模优秀论文\\2017\\Problems\\B\\附件一：已结束项目任务数据.xls'))
#选择有用的数据
data=data[['任务gps 纬度','任务gps经度','任务标价']]
x=data.iloc[:,:2].values.reshape(-1,2) #自变量是前两维,同时要进行reshape
y=data.iloc[:,-1].values.reshape(-1,1)

#初始化核函数(注意对于多维数据，核函数需要更新，可以多试试)
kernel=(ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4))+DotProduct()+WhiteKernel())*RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
#定义高斯过程回归模型
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#＃使用参数的最大似然估计来拟合数据
gpr.fit(x,y)
print(gpr.score(x,y)) #得出回归方程的R方
x_test=np.linspace(np.min(x,axis=0),np.max(x,axis=0),200) #这里生成二维的200个测试数据，注意不能设置太多，不然处理速度很慢且很卡
mu,cov=gpr.predict(x_test,return_cov=True)
y_test=mu.ravel() #将均值向量展平
uncertainty = 1.96 * np.sqrt(np.diag(cov)) #计算95%置信区间，用于后续绘图
X,Y=np.meshgrid(x_test[:,0],x_test[:,1]) #创建和X和Y的网格矩阵
Z=np.ones((200,200)) #为了将Z轴也网格化，先创建全是1的200*200网格
Z=y_test*Z

#plt.style.use('') #目前可改变3D图像样式的有ggplot、seaborn、Solarize_Light2、grayscale
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,6),facecolor='#ffffff')
ax = fig.gca(projection='3d')
#绘制表面图
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
surf = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm,rstride=4,cstride=4,
                       linewidth=0.1, antialiased=False)
#下面是进行三维的三角型拟合得出最终的三维图形
"""surf=ax.plot_trisurf(data.iloc[:,0],data.iloc[:,1],z,
                cmap=cm.coolwarm, edgecolor='none')"""
#绘制等高线
"""cset1 = ax.contourf(X, Y, Z, zdir='z', cmap=cm.coolwarm)
cset2= ax.contourf(X, Y, Z, zdir='x' ,cmap=cm.coolwarm)
cset3 = ax.contourf(X, Y, Z, zdir='y', cmap=cm.coolwarm)
"""
# Add a color bar which maps values to colors.(这里加上颜色条，同时设置其参数)
fig.colorbar(surf, shrink=0.5, aspect=5)

#下面将散点绘制到图片中去
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],marker='o',c=data.iloc[:,2],cmap=cm.coolwarm)

#打上x、y、z轴的标签和标题
ax.set_xlabel('X Label',fontsize=13)
ax.set_ylabel('Y Label',fontsize=13)
ax.set_zlabel('Z Label',fontsize=13)
ax.set_title('3D Fitted Surface Plot',fontsize=15)
plt.show()
