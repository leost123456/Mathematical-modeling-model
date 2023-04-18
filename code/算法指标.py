import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#均方误差MSE
def mean_square_error(y1,y2): #y1是预测值序列，y2是真实值序列
    return np.sum((np.array(y1)-np.array(y2))**2)/len(y1)

#均方根误差（RMSE）
def root_mean_squard_error(y1,y2): #其中有y1是预测序列，y2是真实序列
    return np.sqrt(np.sum(np.square(np.array(y1)-np.array(y2)))/len(y1))

#平均绝对误差（MAE）
def mean_absolute_error(y1,y2):#其中y1是预测序列，y2是真实序列
    return np.sum(np.abs(np.array(y1)-np.array(y2)))/len(y1)

#AIC准则（越小越好）
def cal_AIC(n,mse,num_params):#其中n是观测数量，mse是均方误差，num_params是模型参数个数
    aic=n*np.log(mse)+2*num_params
    return aic

#BIC准则（越小越好，表示模型性能越好和复杂度越低）
def cal_BIC(n,mse,num_params):#其中n是观测数量，mse是均方误差，num_params是模型参数个数
    bic=n*np.log(mse)+num_params*np.log(n)
    return bic

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

#第二种计算R方的方式 (1-(SSE/SST))
def R_Squre2(y_pred,y_real): #其中y_pred是预测值，y_real是真实值,两者都是序列
    y_real_mean = np.mean(y_real)
    y_pred_var=0 #计算预测值减真实值的平方和
    y_real_var=0
    for i in range(len(y_pred)):
        y_pred_var+=(y_pred[i]-y_real[i])**2
        y_real_var+=(y_real[i]-y_real_mean)**2
    return 1-y_pred_var/y_real_var

