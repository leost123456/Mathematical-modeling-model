import numpy as np
#熵权法
def Entropy_weight_method(data): # data是输入的csv文件，输出是权值序列
    yij = data.apply(lambda x: x / x.sum(), axis=0)  # 第i个学生的第j个指标值的比重yij = xij/sum(xij) i=(1,m)
    K = 1 / np.log(len(data))   #常数
    tmp = yij * np.log(yij)
    tmp = np.nan_to_num(tmp)
    ej = -K * (tmp.sum(axis=0))     # 计算第j个指标的信息熵
    wj = (1 - ej) / np.sum(1 - ej)  # 计算第j个指标的权重
    return wj #输出权值序列