import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from sklearn import metrics
from sklearn.preprocessing import label_binarize #用于标签二值化
import seaborn as sns

#解决第一个和第二个问题：1给定观测序列发生概率（可用于评价模型）2进行预测给定观测序列的隐含状态，,并且注意此时的观测状态是离散的
#设置隐含变量的类别(箱子的类别)
states = ["box 1","box 2","box3"]
n_states=len(states) #类别数

#设定观察状态的集合（取出球的颜色）
observations = ["red","white"]
n_observations = len(observations) #类别数

# 设定初始状态分布
start_probability = np.array([0.2, 0.4, 0.4])

#设定状态转移概率分布矩阵（有n种隐含状态，矩阵的形状就是n*n）
transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]])

# 设定观测状态概率矩阵 (有n种隐含状态，m种观测状态，其矩阵形状就是n*m)
emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]])

#下面进行设置模型的参数
model1=hmm.CategoricalHMM(n_components=n_states)
model1.startprob_=start_probability  # 初始状态分布
model1.transmat_=transition_probability  # 状态转移概率分布矩阵
model1.emissionprob_=emission_probability  # 观测状态概率矩阵

#下面进行设定未来的一个观测序列(利用维比特算法)
seen=np.array([0,1,0]).astype(int).reshape(-1,1) #注意要转化成二维的形式（如果观测序列是一维的话）
logprob1,pred_state1=model1.decode(seen,algorithm='viterbi') #得到的第一个参数就是求出log下的序列出现概率（后续要进行还原），第二个参数就是预测的隐含状态序列
print(np.exp(logprob1)) #输出观测序列出现的概率，完成问题1
print(pred_state1) #输出预测的隐含状态,完成问题2

#利用前向算法计算
pred_state2=model1.predict(seen) #得到的第一个参数就是求出log下的序列出现概率（后续要进行还原），第二个参数就是预测的隐含状态序列
print(np.exp(model1.score(seen))) #输出观测序列出现的概率，完成问题1
print(pred_state2) #输出预测的隐含状态,完成问题2

#下面进行参数估计（事先不知道隐含状态的初始概率、状态转移矩阵、输出矩阵），最终得出（生成式模型（无监督学习）和监督学习的方式）
#设置隐含变量的类别(箱子的类别)
states1 = ["box 1","box 2","box3"]
n_states1=len(states1) #类别数

#设定观察状态的集合（取出球的颜色）
observations1 = ["red","white"]
n_observations1 = len(observations1) #类别数

#下面进行设置数据集(注意面对离散的观测变量目前只能用生成式（不能进行有监督学习）)
X2 = np.array([[0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,1,1,1,0,1]])#自定义观测序列
#y  = np.array([[0,1,2,1,2,0,1,0,0,2,1,1,2,1,2,1,2,1,1,2,0,0,1,1,2,1,2,1,0,1]]) #隐含状态（分类标签）

#下面进行设计模型
model2=hmm.CategoricalHMM(n_components=n_states1,n_iter=20,tol=0.01) #n_components就是隐含状态的种类数，n_iter就是迭代的次数,tol就是停止的一个阈值

#下面进行模型训练
model2.fit(X2) #进行模型训练（参数学习）注意此时可以是无监督的参数学习（生成式，感觉也可以用于聚类）

#下面进行输出（一些参数）
print( model2.startprob_) #隐含变量初始概率分布
print( model2.transmat_) #转移概率矩阵 (n*n)
print (model2.emissionprob_) #输出概率矩阵 (n*m)
print (np.exp(model2.score(X2))) #序列出现的概率（似然估计）

#设置测试集数据
X_test=np.array([0,1,1,0,0,1]) #观测序列
y_test=np.array([2,1,0,2,1,1]) #隐含序列
bina_y_test=label_binarize(y_test,classes=[0,1,2]) #二值化后的标签数据，用于后续绘制ROC曲线

#下面进行预测（分类操作）
y_pred=model2.predict(X_test.reshape(-1,1)) #进行隐含变量的预测（对新的观测序列）
print(y_pred)

#下面进行模型的评价
#1首先是分类的精度
print(metrics.accuracy_score(y_test,y_pred))

#2绘制ROC曲线，AUC面积
prob_matrix=model2.predict_proba(X_test.reshape(-1,1)) #注意输出的一个一个矩阵类型

#下面利用ROC曲线进行评价
fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(n_states1): #遍历类别
    fpr[i],tpr[i], _ = metrics.roc_curve(bina_y_test[:, i], prob_matrix[:, i]) #计算x轴和y轴的值
    roc_auc[i] = metrics.auc(fpr[i], tpr[i]) #计算auc面积值
# Compute micro-average ROC curve and ROC area（方法二:将每个类别原始值和预测值都进行展平再进行计算ROC）
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(bina_y_test.ravel(), prob_matrix.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

#绘制ROC曲线
plt.figure(figsize=(8,6))
#绘制平均的ROC曲线
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue'] #颜色序列有几类用几种
for i ,color in enumerate(colors):
    plt.plot(fpr[i],tpr[i],color=color,lw=2,label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i])) #绘制曲线同时打上AUC面积标签
plt.plot([0, 1], [0, 1], 'k--', lw=2)  #绘制对角线
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC-Kurve',fontsize=15)
plt.legend(loc="lower right")
plt.show()

#下面还可以用混淆矩阵进行操作
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap='YlGnBu_r',fmt="d",annot=True)
plt.xlabel('Prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('Real category',fontsize=13)       #数轴是真实类别
plt.title('Confusion matrix',fontsize=15)
plt.show()

#下面进行构建观测变量是多维的并且还是连续数值类型的（利用GMMHMM）并且进行无监督学习（聚类和嗯）和监督学习
# 定义观察序列和标记序列（表示x一共有3个特征列，10个观测值）
X = np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.9, 0.8, 0.7], [0.2, 0.3, 0.4], [0.5, 0.4, 0.3],
              [0.8, 0.7, 0.6], [0.3, 0.4, 0.5], [0.6, 0.5, 0.4], [0.9, 0.8, 0.7], [0.2, 0.3, 0.4]])
y = np.array([0,1,1,0,1,1,0,1,1,0]) #原始标签（隐含状态）

#下面式测试的观测序列
X_test1=np.array([[0.2,0.4,0.6],[0.5,0.2,0.1],[0.1,0.4,0.6]]) #测试数据

# 定义HMM模型
model3 = hmm.GaussianHMM(n_components=2, covariance_type="full",n_iter=100)

# 使用标记序列进行训练
model3.fit(X)

#下面进行预测操作
print('似然为',model3.score(X_test1)) #输出观测序列的最大似然（AIC），可以用于检验模型的精度，选择适合的隐含状态变量
y_pred=model3.predict(X_test1)
y_prob=model3.predict_proba(X_test1) #概率分布预测
print(y_pred)

#下面可以根据模型生成类似的观测值序列(分别包括观测状态序列和隐含状态序列)
print(model3.sample(100)[0]) #观测状态序列
print(model3.sample(100)[1]) #隐含状态序列

#如果想进行一定程度的监督学习的话，应该还可以直接设置模型的参数(从原始的人工计算)
"""model3.startprob_ = startprob #初始概率分布
model3.transmat_ = transmat #转换概率矩阵
model3.means_ = means #每个类别的每个特征的均值（最好设置）n*m矩阵形状（n为隐含状态种类数，m为特征数）
model3.covars_ = covars """