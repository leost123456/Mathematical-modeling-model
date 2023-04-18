import numpy as np
def vector_normalization1(X): #向量标准化就是一种占比（0，1），主要用于TOPSIS，X是输入的矩阵，输出是经过标准化后的矩阵
    for i in range(X.shape[1]):
        X[:,i]=X[:,i]/(sum(X[:,i]**2)**0.5)
    return X

def vector_normalization2(X): #向量标准化方式
    a=[i**2 for i in X]
    return [i/(sum(a)**0.5) for i in X]

#TOPSIS法(结合权重)
def TOPSIS_method(X,w): #输入是待评价的矩阵和权重列表，包含评价指标和各个项目。输出是各个项目对应的相对接近度，其越大则说明越好
    x_max=np.max(X,axis=0) #每个指标的最大值
    x_min=np.min(X,axis=0) #每个指标的最小值
    result=[]
    for i in range(X.shape[0]):
        D1 = sum(((X[i,:]-x_max)**2)*np.array(w))**0.5
        D2= sum(((X[i,:]-x_min)**2)*np.array(w))**0.5
        result.append(D2/(D1+D2))
    return result