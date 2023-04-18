import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pykrige.ok import OrdinaryKriging

data=pd.read_csv('D:\\0\\其他\\单子1数据处理\\测试数据.csv',encoding='gbk')
data=data[['情感因子','行为因子','适切因子','认知因子']]

#一维数据插值（还有其余牛顿插值或者拉格朗日等等）
data['认知因子'].interpolate(method='linear',inplace=True) #一次线性插值法（对开头几个数据没办法填充）
data['认知因子'].interpolate(method='quadratic',inplace=True) #二次插值
data['认知因子'].interpolate(method='cubic',inplace=True) #三次插值

#牛顿插值法(可以插组内的也可以插组外的)（可以用于任何数据的插值，但是缺点也是有的，具体见文档）
x=np.arange(len(data['认知因子'])) #设置x
#主体的函数，xx 和 y 分别是已知的数据点序列，x0是需要求解的插值点
def newton_interpolation(x, y, x0):
    n = len(x)
    f =np.zeros((n,n)) #创建一个存储多阶差商的矩阵
    for i in range(n):
        f[i][0] = y[i] #将第一列的值为原始值
    #下面根据差商的性质然后进行递推
    for j in range(1, n):
        for i in range(j, n):
            f[i][j] = (f[i][j - 1] - f[i - 1][j - 1]) / (x[i] - x[i - j])
    #最后一阶的最后一个数
    res = f[n - 1][n - 1]
    #y用逆推公式求解出x0的函数值
    for i in range(n - 2, -1, -1):
        res = f[i][i] + (x0 - x[i]) * res
    return res
#下面进行输出结果
print(newton_interpolation(x,data['认知因子'],3))

#拉格朗日插值法（也是几乎适用于所有数据，得到的多项式与牛顿插值法类似，在数据点上的值相同但是在其他点的数值不一定相同）
def lagrange(x, y, num_points, x_test): #其中x,y就是原始数据点，num_points就是数据点数量，x_test就是需要插值点的x坐标
    # 所有的基函数值，每个元素代表一个基函数的值
    l = np.zeros(shape=(num_points, ))
    # 计算第k个基函数的值
    for k in range(num_points):
        # 乘法时必须先有一个值
        # 由于l[k]肯定会被至少乘n次，所以可以取1
        l[k] = 1
        # 计算第k个基函数中第k_个项（每一项：分子除以分母）
        for k_ in range(num_points):
            # 这里没搞清楚，书中公式上没有对k=k_时，即分母为0进行说明
            # 有些资料上显示k是不等于k_的
            if k != k_:
                # 基函数需要通过连乘得到
                l[k] = l[k]*(x_test-x[k_])/(x[k]-x[k_])
            else: #如果分母为0直接跳过
                pass
    # 计算当前需要预测的x_test对应的y_test值
    L = 0
    for i in range(num_points):
        # 求所有基函数值的和
        L += y[i]*l[i]
    return L
#下面进行输出结果
test_x=np.linspace(0,100,1000) #生成1000个测试样本点
interpolate_data=[lagrange(x,data['认知因子'],len(x),k) for k in test_x] #得到所有测试样本点的插值
print(lagrange(x,data['认知因子'],len(x),3))

#样条插值法（在不同的x轴区间用不同的插值函数（多项式）进行操作）结合前面的就行

#二维的插值方法（可以参考可视化模块中的”插值三维可视化.py文件“）

#下面是进行克里金空间插值（适用空间分布、时空分布的情况）
#读取数据
data1=pd.read_csv('D:\\2022国赛数模练习题\\2022宁波大学数学建模暑期训练第四轮\\附件1.csv',encoding='gbk')
data2=pd.read_csv('D:\\2022国赛数模练习题\\2022宁波大学数学建模暑期训练第四轮\\附件2.csv',encoding='gbk')
new_data=pd.concat((data1,data2),axis=1) #进行数据组合
new_data.drop(columns=['编号'],inplace=True) #去掉无用的列，目前剩下12列

#下面进行分出各个区域，用于后续在云图中绘制散点
sheng_data=new_data[new_data['功能区']==1] #生活区
gong_data=new_data[new_data['功能区']==2]  #工业区
shan_data=new_data[new_data['功能区']==3]  #山区
jiao_data=new_data[new_data['功能区']==4]  #交通区
yuan_data=new_data[new_data['功能区']==5]  #公园

#data1是x轴，data2是y轴，data3是z轴数据，title是图的标题，xlabel是x轴标题，ylabel是y轴标题，ctitle是颜色条的标题，file_name是存储图的名称
def kriging_interpolation(data1,data2,data3,title,xlabel,ylabel,ctitle,file_name):
    #先生成等长的x,y序列
    grid_x=np.linspace(np.min(data1),np.max(data1),300)
    grid_y=np.linspace(np.min(data2),np.max(data2),300)
    #创建克里金插值的对象，其中的variogram_model表示在克里金插值中使用高斯变异函数（注意还可以设置其他函数，具体看文档），并通过nlags=6 设置了变异函数的参数。
    OK=OrdinaryKriging(data1,data2,data3,variogram_model='gaussian',nlags=6)
    z1,ss1=OK.execute('grid',grid_x,grid_y)
    X,Y=np.meshgrid(grid_x,grid_y)
    plt.figure(figsize=(8,6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #绘制云图
    levels=range(round(np.min(data3)),round(np.max(data3)),3) #设置渐变范围
    a1=plt.contourf(X,Y,z1, levels, cmap=cm.Spectral, alpha=0.7)
    #绘制等高线
    a2=plt.contour(X,Y,z1,linestyles='solid',linewidth=1)
    # 加入收集点的坐标
    plt.scatter(sheng_data['x(m)'], sheng_data['y(m)'], marker='o', label='生活区')
    plt.scatter(gong_data['x(m)'], gong_data['y(m)'], marker='s', label='工业区')
    plt.scatter(shan_data['x(m)'], shan_data['y(m)'], marker='+', label='山区')
    plt.scatter(jiao_data['x(m)'], jiao_data['y(m)'], marker='D', label='交通区')
    plt.scatter(yuan_data['x(m)'], yuan_data['y(m)'], marker='^', label='公园绿地区')
    plt.legend(loc='upper left')
    cbar=plt.colorbar(a1) #为云图添加颜色条
    cbar.set_label(ctitle, fontsize=13)  # 设置颜色条的标题
    plt.clabel(a2,inline=True,fontsize=15) #为等高线添加标签
    plt.title(title,fontsize=15)
    plt.xlabel(xlabel,fontsize=13)
    plt.ylabel(ylabel,fontsize=13)
    #plt.savefig(file_name,format="svg",bbox_inches='tight')
    plt.show()
kriging_interpolation(new_data['x(m)'],new_data['y(m)'],new_data['As (μg/g)'],'As的空间分布','x(m)','y(m)','浓度(μg/g)','As.svg')