import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize

#标准化功能函数(处理矩阵)
def standardization(data): #data是矩阵,主要功能是进行标准化,输出是经过标准化的矩阵
    data_std=[np.std(data[:,i]) for i in range(data.shape[1])]
    data_mean=[np.mean(data[:,i]) for i in range(data.shape[1])]
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-data_mean[i])/data_std[i]
    return data

#标准化功能函数(处理序列)
def standardization2(data): #data是序列,主要功能是进行标准化,输出是经过标准化的序列
    data_std=np.std(data)
    data_mean=np.mean(data)
    for i in range(len(data)):
        data[i]=(data[i]-data_mean)/data_std
    return data

#进行01标准化操作
def Normalized1(data): #data是csv,这个是极大型指标的处理方式 输出也是序列。
    scaler= MinMaxScaler()
    return scaler.fit_transform(data)

#进行极大指标值正向化操作（0，1） 注意会有0的出现，再之后的代码中如果除数为0就会出bug
def Normalized2(data): #data是输入序列,这个是极大型指标的处理方式 输出也是序列
    min_data=min(data)
    max_data=max(data)
    return [(x-min_data)/(max_data-min_data) for x in data]

#极小型指标指标的正向化方法
def mindata(data):
    max_num=max(data)
    min_num=min(data)
    result=[(max_num-x)/(max_num-min_num) for x in data]
    return result

#将中间型指标正向化的方法
def middata(data): #data是待正向化的序列，输出是经过正向化的序列
    max_num=max(data)
    min_num=min(data)
    result=[]
    for x in data:
        if x>=min_num and x<=((max_num+min_num)/2):
            result.append(2*(x-min_num)/(max_num-min_num))
        elif x>=((max_num+min_num)/2) and x<=max_num:
            result.append(2*(max_num-x)/(max_num-min_num))
    return result

#区间型指标正向化的方法
def interval_data(data,a,b): #data是输入序列，其中a是下界，b是上界，输出是经过正向化的序列
    min_data=min(data)
    max_data=max(data)
    c=max([a-min_data,max_data-b])
    result=[]
    for x in data:
        if x<a:
            result.append(1-((a-x)/c))
        elif x>=a and x<=b:
            result.append(1)
        elif x>b:
            result.append(1-((x-b)/c))
    return result

#下面的向量标准化并不会出现0的情况，是一种占比
def vector_normalization1(X): #向量标准化就是一种占比（0，1），主要用于TOPSIS，X是输入的矩阵，输出是经过标准化后的矩阵
    for i in range(X.shape[1]):
        X[:,i]=X[:,i]/(sum(X[:,i]**2)**0.5)
    return X

def vector_normalization2(X): #向量标准化方式，输入是一个序列，输出也是一个序列
    a=[i**2 for i in X]
    return [i/(sum(a)**0.5) for i in X]

#下面是进行one-hot编码的函数
def one_hot(data,classes): #data是输入数据，序列的形式，classes是类别名称列表,例如['A','B','C']
    return label_binarize(data,classes=classes) #输出的是多维矩阵的形式，每一个类别都是0/1变量