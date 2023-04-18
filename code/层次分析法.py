import pandas as pd
import numpy as np
import warnings

#下面进行构建层次分析法的对象(注意其准则层最多14个，方案层也最多14个，不然无法计算RI)
class AHP: #注意其中算法包括'arithmetic mean'(算术平均) 'geometric mean'(几何平均) ,'Eigenvalues'（特征值法） 'comprehensive'(综合法)
    def __init__(self, criteria, b=None,algorithm='comprehensive'): #其中criteria表示准测层的重要性矩阵，b表示方案层的重要性矩阵（对每个准则）列表，可以不用,最后一个表示计算权重的算法
        self.RI = (0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49,1.52,1.54,1.56,1.58,1.59) #RI的查询列表，用于后续计算一致性比例CR
        self.criteria = criteria
        self.algorithm = algorithm  # 存储权重算法
        self.b = b
        #下面进行判断准则层和方案层是否输入为空，如果为空则是一般需要只需要计算输入的重要性矩阵的权重
        self.num_criteria = criteria.shape[0] #统计准则层元素的个数
        #如果只想计算搞准则层（指标层）的权重，就不用设置方案层的重要性矩阵了
        if b!=None:
            self.num_project = b[0].shape[0] #统计方案个数

    #下面的函数是用于计算权重的，输入为重要性矩阵，输出为权重的列表向量
    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)

        #下面进行保证其是个方阵
        n, n1 = input_matrix.shape
        assert n == n1, '不是一个方阵'

        #下面进行判断此方阵是否为正互反矩阵（要正互反矩阵才行）
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反互对称矩阵')

        #下面得到重要性矩阵的特征值序列和特征向量矩阵
        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)  #eigenvalues为特征值序列，eigenvectors为特征向量矩阵（竖着看）
        max_idx = np.argmax(eigenvalues)  # 计算最大特征值
        max_eigen = eigenvalues[max_idx].real  # 得到最大特征值的实部（原来是个复数）

        #1第一种算术平均法求权重
        weights1=np.sum(input_matrix/np.sum(input_matrix,axis=0),axis=1)/n

        #2第二种几何平均法求权重
        weights2=input_matrix[:,0] #先取第一列
        #先按行相乘得到新的列向量
        for i in range(1,n):
            weights2=weights2*input_matrix[:,i]
        weights2=(weights2**(1/n))/np.sum(weights2**(1/n))

        # 3下面是利用特征值法求权重
        max_eigenvectors = eigenvectors[:, max_idx].real #得到最大特征向量列的实部
        weights3 = max_eigenvectors / max_eigenvectors.sum() #得到权重序列

        #4综合上述方法得到一个综合的权重
        weights4=(weights1+weights2+weights3)/3

        #下面进行使用的权重算法判断
        if  self.algorithm == 'arithmetic mean':
            weights=weights1
        elif self.algorithm == 'geometric mean':
            weights=weights2
        elif self.algorithm == 'Eigenvalues':
            weights=weights3
        elif self.algorithm == 'comprehensive':
            weights = weights4

        #下面进行计算一致性比例CR，注意当元素数量大于14个无法判断
        if n > 14:
            CR = None
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n-1]

        return max_eigen, CR, weights #返回最大特征值、一致性比例（CR）和权重序列

    #主体运行函数
    def run(self):
        #下面得到准则层的最大特征值、一致性比例（CR）和权重序列
        max_eigen, CR, criteria_weights = self.cal_weights(self.criteria)
        #下面进行输出信息，注意CR<0.1认为重要性矩阵的一致性通过
        print('准则层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_weights))

        #下面得到所有方案层的准则层的最大特征值、一致性比例（CR）和权重评分序列（均存储在列表中）
        max_eigen_list, CR_list, weights_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            weights_list.append(eigen)

        #注意输出的结构是column是各个方案，index是各个准则（特征）
        pd_print = pd.DataFrame(np.array(weights_list),
                                index=['准则' + str(i) for i in range(self.num_criteria)],
                                columns=['方案' + str(i) for i in range(self.num_project)],)
        #下面对dataframe添加各个准则下的方案层的最大特征值（由方案层得出）、CR、和一致性检验结果
        pd_print.loc[:, '最大特征值'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1
        print('方案层权重评分和其他信息')
        print(pd_print)

        # 目标层（下面进行得到综合的评分，通过直接全部权重相乘的形式（准则层和方案层））
        obj = np.dot(criteria_weights.reshape(1, -1), np.array(weights_list))
        print('\n目标层的综合评分', obj)
        print('最优选择是方案{}'.format(np.argmax(obj)))
        return obj

if __name__ == '__main__':
    # 准则重要性矩阵 （总共有5个特征（准则））
    criteria = np.array([[1, 2, 7, 5, 5],
                         [1 / 2, 1, 4, 3, 3],
                         [1 / 7, 1 / 4, 1, 1 / 2, 1 / 3],
                         [1 / 5, 1 / 3, 2, 1, 1],
                         [1 / 5, 1 / 3, 3, 1, 1]])

    # 对每个准则，方案优劣排序（总共有3个方案）
    b1 = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    b2 = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    b3 = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
    b4 = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    b5 = np.array([[1, 4, 1 / 2], [1 / 4, 1, 1 / 4], [2, 4, 1]])
    # 方案层重要性列表（对每个准则）
    b = [b1, b2, b3, b4, b5]

    #最终输出
    a = AHP(criteria,b,'comprehensive').run()
    #如果只想要知道某个重要性矩阵的权重（比如只需要准则层（特征层）的权重）
    max_eigen,CR,weights = AHP(criteria).cal_weights(criteria) #其中max_eigen就是最大特征根，CR是一致性比例，weights就是权重序列
    print(weights)
