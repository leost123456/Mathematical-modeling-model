import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

#读取数据（注意数据都已经正向化过了）
df=pd.read_csv('D:\\2022国赛数模练习题\\有风险系数的有信贷信息企业.csv',encoding='gbk')
#展现数据
#print(df.head())

#定于需要数据的column
data_columns=['是否违约','信誉评级','合格发票率','年平均利润','净发票总金额','总盈利率','月平均增长盈利率','风险因子','风险系数']
data=df[data_columns] #选取有用的数据
#简答的看一下数据的总体情况
#print(df.describe())
"""plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df.hist()#等于是创建了一个画板并且在里面已经绘制了东西
plt.show()"""

# 需要自行添加逻辑回归所需的intercept变量
data['intercept'] = 1.0

#指定作为训练变量的列，不含目标（注意可以用sklearn中的train_test_split进行分割训练集和测试集）
train_cols=data_columns[1:]
#定义logit模型
logit=sm.Logit(data['是否违约'],data[train_cols])
#拟合模型
result=logit.fit()
#最终查看数据要点
print(result.summary()) #可以看伪R方来看拟合的程度
#下面可以进行预测操作
# 构建预测集
# 与训练集相似，一般也是通过 pd.read_csv() 读入
# 在这边为方便，我们将训练集拷贝一份作为预测集（不包括 admin 列）
combos = copy.deepcopy(data)
# 预测集也要添加intercept变量
combos['intercept'] = 1.0
combos['predict'] = result.predict(combos[data_columns[1:]])
# 预测完成后，predict 的值是介于 [0, 1] 间的概率值
# 我们可以根据需要，提取预测结果
# 例如，假定 predict > 0.5，则表示会被录取
# 在这边我们检验一下上述选取结果的精确度
combos['预测是否违约']=combos['predict'].apply(lambda x:1 if x>0.5 else 0)
#计算精确度
print(1-sum(np.abs(combos['预测是否违约']-data['是否违约']))/len(combos['预测是否违约']))

#第二种方式(注意sklearn中的逻辑回归模型会自动进行正则化，因此求出的参数跟sm中的模型不一致)
#其中penalty是惩罚项，可以选择l1或l2正则化，是防止过拟合用的。solver是优化方法有五种类型。最后的C表示正则化强度的倒数，必须是一个大于0的浮点数，不填写默认1.0，即默认正则项与损失函数的比值是1：1。C越小，损失函数会越小，模型对损失函数的惩罚越重，正则化的效力越强，参数会逐渐被压缩得越来越小。
logit1= LogisticRegression(fit_intercept = False)  #其中penalty是惩罚项，可以选择l1或l2正则化，是防止过拟合用的。solver是优化方法有五种类型。最后
#penalty="l2",solver="liblinear",C=0.5,max_iter=1000 参数设置
logit1.fit(data[train_cols], data['是否违约']) #进行训练模型
print(logit1.coef_) #输出逻辑回归方程系数
print(logit1.score(data[train_cols],data['是否违约'])) #得到准确率
#logit1.predict(x_test) #进行预测
logit1_predict=logit1.predict(data[train_cols])     #预测标签
logit1_proba=logit1.predict_proba(data[train_cols])[:,1] #计算二分类概率(计算其为1的概率)

#下面我们进行可以进行绘制ROC曲线和计算AUC值（这里是计算总体的）注意需要用概率来进行计算
fpr,tpr,_=metrics.roc_curve(combos['是否违约'],combos['predict']) #第一个参数是原始值，第二个参数是预测的概率值，计算ROC曲线点值
roc_auc =metrics.auc(fpr,tpr)
#绘制ROC曲线
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr,label='ROC curve (area = {0:0.2f})'''.format(roc_auc),
         color='deeppink', linewidth=4)
plt.stackplot(fpr,tpr,colors='steelblue',alpha=0.5,edgecolor='black')
plt.plot([0, 1], [0, 1], 'k--', lw=2) #绘制对角线
plt.text(0.5,0.3,f'ROC curve(area = {roc_auc:.4f})',fontsize=13) #打上标签
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC-Kurve',fontsize=15)
plt.show()

#下面进行绘制混淆矩阵
confusion_matrix=metrics.confusion_matrix(combos['是否违约'],logit1_predict)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
plt.xlabel('Prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('Real category',fontsize=13)       #数轴是真实类别
plt.title('Confusion matrix',fontsize=15)
plt.show()

#下面进行一次性得到多分类的逻辑回归参数，并绘制多分类的ROC曲线
new_data=pd.read_csv('D:\\2023美赛\\2023\\带有标签的数据.csv',encoding='gbk')
X=new_data[['same_letter','num_vowel','num_monogram']].values #自变量数据
# 创建多类被逻辑回归对象（有几个分类标签就创建几个逻辑回归模型）
clf = LogisticRegression(multi_class='ovr')
# 拟合模型
clf.fit(X,new_data['label'])
# 计算系数（输出所有逻辑回归模型的回归系数矩阵 ）
coef = clf.coef_

#下面进行绘制多类别的ROC曲线（可以采用下面的方式，可以自己通过搭建多个逻辑回归模型搞）
Y=label_binarize(new_data['label'], classes=[0, 1, 2]) #进行标签二值化
n_classes = Y.shape[1]
# 创建逻辑回归对象
clf = OneVsRestClassifier(LogisticRegression())
# 训练模型
y_score = clf.fit(X, Y).decision_function(X)

# 计算每个类别的ROC曲线和AUC值
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#下面是计算均值
fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#开始绘制ROC曲线
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
#绘制平均的ROC曲线
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = ['#2779ac','#f2811d', '#349d35'] #颜色序列有几类用几种
for i ,color in enumerate(colors):
    plt.plot(fpr[i],tpr[i],color=color,lw=2,label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i])) #绘制曲线同时打上AUC面积标签
plt.plot([0, 1], [0, 1], 'k--', lw=2)  #绘制对角线
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC-Kurve',fontsize=15)
plt.legend(loc="lower right")
#plt.savefig('D:\\2023美赛\\2023\\图\\ROC曲线.svg',format='svg',bbox_inches='tight')
plt.show()

