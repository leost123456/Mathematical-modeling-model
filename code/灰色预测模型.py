import numpy as np
import pandas as pd

#下面进行构造级比检验的函数
def scale_test(data,n): #data为输入的序列,n为序列元素个数
    left_value=np.exp(-(2/(n+1))) #左区间
    right_value=np.exp(2/(n+1))  #右区间
    new_data1=np.array(data[:-1])+1000 #转化为array类型,取前n-1个
    new_data2=np.array(data[1:])+1000 #取后n-1个数据
    result=new_data1/new_data2
    print(min(result),max(result))
    print(left_value)
    print(right_value)
    return result,left_value,right_value\

"""scale_test(new_EA_Y[:-3],14)
scale_test(new_WA_Y[:-3],14)
scale_test(new_AS_Y[:-3],14)
scale_test(new_MS_Y[:-3],14)"""

#下面进行构造灰色预测模型的函数

def GM11(x, n):
    '''
    灰色预测
    x：序列，numpy对象
    n:需要往后预测的个数
    '''
    x1 = x.cumsum()  # 一次累加
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x[1:].reshape((len(x) - 1, 1))
    # a为发展系数 b为灰色作用量
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)  # 计算待估参数
    result = (x[0] - b / a) * np.exp(-a * (n - 1)) - (x[0] - b / a) * np.exp(-a * (n - 2))  # 预测方程
    S1_2 = x.var()  # 原序列方差
    e = list()  # 残差序列
    for index in range(1, x.shape[0] + 1):
        predict = (x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2))
        e.append(x[index - 1] - predict)
        print(predict)  # 预测值
    print("后验差检验")
    S2_2 = np.array(e).var()  # 残差方差
    C = S2_2 / S1_2  # 后验差比
    if C <= 0.35:
        assess = '后验差比<=0.35，模型精度等级为好'
    elif C <= 0.5:
        assess = '后验差比<=0.5，模型精度等级为合格'
    elif C <= 0.65:
        assess = '后验差比<=0.65，模型精度等级为勉强'
    else:
        assess = '后验差比>0.65，模型精度等级为不合格'
    # 预测数据
    predict = list()
    for index in range(x.shape[0] + 1, x.shape[0] + n + 1):
        predict.append((x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2)))
    predict = np.array(predict)

    return a,b,predict,C,assess #其中a为发展系数，b为灰色作用量，predict为预测的序列，C为后验差比，assess是后验差比结构

#a,b,predict,C,assess=GM11(np.array(new_AS_Y[:-3])+1000,3)
