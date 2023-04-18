import random
import numpy as np

#下面是随机生成一定数量的符合正态分布的数据
def randomtest_num(mu,sigma,num): #mu是均值，sigma是标准差，num是生成的数量
    data=[]
    for i in range(num):
        data.append(random.normalvariate(mu=mu,sigma=sigma))
    return data

#生成均匀分布的点
def Evenly_distributed(a,b,num): #a是起始位置，b是结束位置，num是生成的个数
    data=[]
    for i in range(num):
        data.append(a+(b-a)*np.random.rand(1))
    return data

#生成符合二项分布的点
d1=np.random.binomial(n=9,p=0.5,size=100)   #n是实验次数，p是概率，size代表生成的个数

#生成超几何分布的点
#对以下代码可以解释为：共50个产品，正品有47个，次品有3个，每次从这些产品中取3个出来，共重复这样的试验50次。假设返回的是每一次所抽取的三个产品中的正品数，共50个值。
d2=np.random.hypergeometric(47,3,3,size=30) #对上述代码可以解释为：共50个产品，正品有47个，次品有3个，每次从这些产品中取3个出来，共重复这样的试验50次。假设返回的是每一次所抽取的三个产品中的正品数，共50个值。

