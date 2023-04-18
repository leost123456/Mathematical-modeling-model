import numpy as np
import pandas as pd
import statistics as stat
from fancyimpute import KNN
import matplotlib.pyplot as plt
import seaborn as sns

#先导入测试的数据（包括4列的变量，其中有缺失值）
data=pd.read_csv('D:\\0\\其他\\单子1数据处理\\测试数据.csv',encoding='gbk')
data=data[['情感因子','行为因子','适切因子','认知因子']]

#1进行重复值处理（其中subset是检查哪些列，keep是有重复值就保留第一个,ignore_index表示是否删除重复值后重新进行排序），注意是直接删除整行
#data.drop_duplicates(subset=['情感因子'],keep='first',inplace=True,ignore_index=True)

#2进行缺失值观测和处理
#print(data.isnull().sum()) #统计各个变量列缺失值的个数

#下面进行缺失值处理
#(1)直接删除对应的行
#data.dropna(inplace=True)

#(2)前后补充法
"""data.fillna(method='ffill',inplace=True) #前向补充法(对于开头几个数据没办法填充)
data.fillna(method='bfill',inplace=True) #后向补充法(对最后的几个数据没办法填充)
"""

#(3)用均值、中位数、众数进行填补
#data['情感因子'].fillna(np.mean(data['情感因子']),inplace=True) #中位数：np.median 众数：stat.mode

#(4)插值法（还有其余牛顿插值或者拉格朗日等等）
#data['情感因子'].interpolate(method='linear',inplace=True) #一次线性插值法（对开头几个数据没办法填充）
#data['情感因子'].interpolate(method='quadratic',inplace=True) #二次插值
#data['情感因子'].interpolate(method='cubic',inplace=True) #三次插值

#(5)MC算法填补缺失值
class MF():
    def __init__(self, X, k, alpha, beta, iterations): #
        """
        Perform matrix factorization to predict np.nan entries in a matrix.
        Arguments
        - X (ndarray)   : sample-feature matrix 输入的矩阵
        - k (int)       : number of latent dimensions 列的维度，k越大计算量越大
        - alpha (float) : learning rate #学习率
        - beta (float)  : regularization parameter #正则化的参数
        - iterations    : 迭代的轮数
        """
        self.X = X
        self.num_samples, self.num_features = X.shape
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        # True if not nan
        self.not_nan_index = (np.isnan(self.X) == False)

    def train(self): #训练的主体函数
        # Initialize factorization matrix U and V 先对U和V矩阵进行初始化
        self.U = np.random.normal(scale=1./self.k, size=(self.num_samples, self.k))
        self.V = np.random.normal(scale=1./self.k, size=(self.num_features, self.k))

        # Initialize the biases 对偏置进行初始化
        self.b_u = np.zeros(self.num_samples)
        self.b_v = np.zeros(self.num_features)
        self.b = np.mean(self.X[np.where(self.not_nan_index)])
        # Create a list of training samples 创建训练样本，注意是选择非空值的
        self.samples = [
            (i, j, self.X[i, j])
            for i in range(self.num_samples)
            for j in range(self.num_features)
            if not np.isnan(self.X[i, j])
        ]

        # Perform stochastic gradient descent for number of iterations 进行梯度下降更新参数
        training_process = []
        for i in range(self.iterations): #一轮一轮迭代
            np.random.shuffle(self.samples) #随机打乱
            self.sgd() #进行梯度下降迭代
            # total square error
            se = self.square_error() #计算loss也就是误差
            training_process.append((i, se)) #存储第几轮和误差
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, se)) #每10轮输出误差
        return training_process

    def square_error(self): #计算误差二次范数
        """
        A function to compute the total square error
        """
        predicted = self.full_matrix()
        error = 0
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index[i, j]:
                    error += pow(self.X[i, j] - predicted[i, j], 2)
        return error

    def sgd(self): #进行梯度下降
        """
        Perform stochastic graident descent
        """
        for i, j, x in self.samples:
            # Computer prediction and error
            prediction = self.get_x(i, j) #得到预测的矩阵
            e = (x - prediction) #求误差（一次范数）

            # Update biases #更新偏置
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])

            # Update factorization matrix U and V #更新U和V矩阵
            """
            If RuntimeWarning: overflow encountered in multiply,
            then turn down the learning rate alpha.
            """
            self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i,:])
            self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j,:])

    def get_x(self, i, j): #得到预测的矩阵（用上U,V,b_u,b_v和b）
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self): #产生最后的矩阵（预测的）训练完后进行生成
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_v[np.newaxis, :] + self.U.dot(self.V.T)

    def replace_nan(self, X_hat): #将原始矩阵中的缺失值进行替换
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X

# np.random.seed(1)
"""mf = MF(data.values, k=2, alpha=0.1, beta=0.1, iterations=100) #创建MF对象
mf.train() #进行训练
X_hat = mf.full_matrix() #得到最后预测的矩阵
X_comp = mf.replace_nan(X_hat) #输出填补缺失值后的原始矩阵
new_data=pd.DataFrame(X_comp,columns=data.columns)"""

#(6)用机器学习算法补全缺失值（例如KNN和随机森林）
#下面是用KNN进行填补
"""fill_knn=KNN(k=3).fit_transform(data) #直接用k近邻算法
new_data=pd.DataFrame(fill_knn,columns=data.columns)"""

#3进行异常值检测与剔除
#绘制箱型图观察并去除异常点（适用于多列数据）
plt.rcParams['font.sans-serif'] = ['SimHei'] #中文显示
plt.rcParams['axes.unicode_minus'] = False
'''plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
f=data.boxplot(sym='r*',return_type='dict',patch_artist=True) #sym表示异常点的形态是红色星号的
for box in f['boxes']:
    # 箱体边框颜色
    box.set( color='#7570b3', linewidth=2)
    # 箱体内部填充颜色
    box.set( facecolor = '#1b9e77' )
#虚线的参数
for whisker in f['whiskers']:
    whisker.set(color='#bf0000',ls='dashed',linewidth=2)
#四分位数线的参数
for cap in f['caps']: #四分位数线的参数
    cap.set(color='g', linewidth=3)
#中位数线的参数
for median in f['medians']:
    median.set(color='DarkBlue', linewidth=3)
#异常点的参数（注意如果想给异常值打上标签的话可以用后面删除异常点的行数据代码进行筛选数据）
for flier in f['fliers']:
    flier.set(marker='*', color='r', alpha=0.5)
plt.xlabel('类别',fontsize=13)
plt.ylabel('数值',fontsize=13)
plt.title('各变量分布情况箱型图',fontsize=15)
plt.show()'''

#下面进行删除异常点的行数据
"""for column in data.columns:
    Q1=data[column].quantile(q=0.25) #下四分位数
    Q2=data[column].quantile(q=0.75)  #上四分位数
    low_whisker = Q1 - 1.5 * (Q2 - Q1)  # 下边缘
    up_whisker = Q2 + 1.5 * (Q2 - Q1)  # 上边缘
    #下面进行筛选
    data=data[(data[column]>=low_whisker)&(data[column]<=up_whisker)]"""

#如果是单列的数据可以绘制散点图加上区间来展现(3倍标准差)
"""upper=np.mean(data['情感因子'])+3*np.std(data['情感因子'])
lower=np.mean(data['情感因子'])-3*np.std(data['情感因子'])
print(upper,lower)
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
plt.scatter(np.arange(len(data['情感因子'])),data['情感因子'],s=15,color='#F21855',alpha=0.9)
#下面进行绘制区间线（注意如果想要对异常点打标签，可以创建一个Dataframe然后进行筛选，然后再绘制（加上特殊形状与颜色和打上标签））
plt.axhline(y=upper, color='#bf0000', linestyle='dashed')
plt.axhline(y=lower, color='#bf0000', linestyle='dashed')
plt.yticks([2.268,2.5,3.0,3.5,4.0,4.5,5.0,5.5,5.887,6.0],[2.3,2.5,3.0,3.5,4.0,4.5,5.0,5.5,5.9,6.0])
plt.xlabel('序号',fontsize=13) #上区间
plt.ylabel('情感因子',fontsize=13) #下区间
plt.title('情感因子数据分布图',fontsize=15)
plt.show()
#下面将异常值进行剔除
data=data[(data['情感因子']>=lower)&(data['情感因子']<=upper)]"""



