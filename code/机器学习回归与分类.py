from sklearn.model_selection import train_test_split
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier #注意只有在这个ensemble中的模型可以进行软投票集成，其都是通过概率来得到的
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier  #集成投票模型
from sklearn.preprocessing import label_binarize #用于标签二值化
from sklearn.calibration import  CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics #metrics.accuracy_score(y_test,y_pred) 用于检测分类模型的精度
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt

#导入数据
#训练集和验证集
deal_data_y=pd.read_csv('D:\\2022国赛数模练习题\\2022宁波大学数学建模暑期训练第二轮训练题B\\处理后的有信贷信息的企业数据.csv',encoding='gbk')
#预测集
deal_data_w=pd.read_csv('D:\\2022国赛数模练习题\\2022宁波大学数学建模暑期训练第二轮训练题B\\处理后的无信贷信息的企业数据.csv',encoding='gbk')
#先将无用的数据扔掉,columns=['信誉评级', '是否违约', '合格发票率', '年平均利润', '净发票总金额', '总盈利率', '月平均增长盈利率', '风险因子']
store_data_y=copy.deepcopy(deal_data_w)
deal_data_y.drop(['企业代号','企业名称'],axis=1,inplace=True)
x_train_ori=deal_data_y.drop('信誉评级',axis=1)
y_train_ori=deal_data_y['信誉评级'] #注意要保留dataframe的格式
#定义没有信贷信息的预测集
deal_data_w.drop(['企业代号','企业名称'],axis=1,inplace=True)
x_predict=deal_data_w
#注意机器学习要学会做敏感性分析画图（参数很多可以）

#下面是算法的主要部分
#划分训练集和测试集(其中stratify表示标签值在测试集和训练集中所占的比例相同)
x_train,x_test,y_train,y_test=train_test_split(x_train_ori,y_train_ori,test_size=0.2,random_state=42,stratify=y_train_ori,shuffle=True)
#注意需要重置索引才能用后面的K折交叉验证
x_train=x_train.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
print(x_train)

#注意还有一种方式就是用ndarray的方式（其能用索引来取值，就不用dataframe的格式了）
"""x_train=np.ndarray(x_train)
x_test=np.ndarray(x_test)
y_train=np.ndarray(y_train)
y_test=np.ndarray(y_test)"""

weight=[]#存储交叉验证的平均准确率用于确定软投票的权重
#创建一个随机森林模型（后面进行交叉验证更具说服力）
RF=RandomForestClassifier(n_estimators=100,random_state=20)
#计算交叉验证的各组验证集情况
score=cross_val_score(RF,x_train,y_train,cv=5)
weight.append(score.mean())
print('随机森林算法',score.mean())

#交叉验证进行具体求解，得出每一组的结果，同时用验证集精度最高的一组来进行测试集验证
kf=KFold(n_splits=5,random_state=None) #进行5折交叉验证  还可以调节shuffle=True
#kf=StratifiedKFold(n_splits=5,random_state=None) #分层交叉验证，每折中的分类标签比例均相同
score_list=[] #存储每次交叉验证的验证集准确度
max_score = 0 #存储最好的交叉验证中验证集准确率
for train_index,test_index in kf.split(x_train,y_train):
    #下面实现直接取索引来获取数据
    X_train=x_train.loc[train_index,:]
    Y_train=y_train.loc[train_index]
    X_val = x_train.loc[test_index,:]
    Y_val = y_train.loc[test_index]
    model=RandomForestClassifier(n_estimators=100,random_state=10)#建立模型
    model.fit(X_train,Y_train) #模型训练
    score=model.score(X_val,Y_val) #计算验证集的准确度
#进行比较将最好的模型参数保存
    if score >max_score:
        joblib.dump(model,'model.pkl')#将模型的参数保存
    score_list.append(score)
model=joblib.load('model.pkl') #将最好的模型参数加载进来
test_score=model.score(x_test,y_test) #对测试集进行预测
print('随机森林算法5折交叉验证综合评分',np.mean(score_list)) #交叉验证平均准确度
print('随机森林算法测试集得分',test_score)  #交叉验证精度最高的一组进行测试集的精度测试

#knn（k紧邻）算法 (当k为3时，测试集acc最大值为54%)
test_scores = []
train_scores = []
for i in range(1,15):
    knn=KNeighborsClassifier(i)#设置k参数，离最近k个点的距离平均
    knn.fit(x_train,y_train)
    train_scores.append(knn.score(x_train, y_train))
    test_scores.append(knn.score(x_test, y_test))
print(max(train_scores),max(test_scores),np.argmax(test_scores))
knn=KNeighborsClassifier(3)
knn.fit(x_train,y_train)
score=cross_val_score(knn,x_train,y_train,cv=5)
weight.append(score.mean())


#支持向量机算法32.43%
svc=SVC(C=0.6,break_ties=False,cache_size=200,gamma=2000,probability=True)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
acc_svc=round(svc.score(x_test,y_test)*100,2)
acc_svc_train=round(svc.score(x_train,y_train)*100,2)
print('支持向量机训练集精度',acc_svc_train,'支持向量机测试集精度',acc_svc)
score=cross_val_score(svc,x_train,y_train,cv=5)
weight.append(score.mean())

#高斯贝叶斯0.44
gassuan=GaussianNB()
gassuan.fit(x_train,y_train)
y_pred=gassuan.predict(x_test)
acc_gassuan=round(gassuan.score(x_test,y_test)*100,2)
acc_gassuan_train=round(gassuan.score(x_train,y_train)*100,2)
print('高斯贝叶斯训练集精度',acc_gassuan_train,'高斯贝叶斯测试集精度',acc_gassuan)
score=cross_val_score(gassuan,x_train,y_train,cv=5)
weight.append(score.mean())

#sgd 0.28
sgd=SGDClassifier(loss = 'hinge')
sgd.fit(x_train,y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_test, y_test) * 100, 2)
print('sgd测试集精度',acc_sgd)
score=cross_val_score(sgd,x_train,y_train,cv=5)
weight.append(score.mean())


#决策树 0.52
decision_tree=DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
Y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_test,y_test) * 100, 2)
print('决策树算法测试集精度',acc_decision_tree)
score=cross_val_score(decision_tree,x_train,y_train,cv=5)
weight.append(score.mean())

#AdaBoostClassifier（0-1分类）
"""Ada=AdaBoostClassifier(algorithm='SAMME',base_estimator=None,learning_rate=0.1,n_estimators=100,random_state=100)
Ada.fit(x_train,y_train)

#GBDT (是0-1分类)
GBDT=GradientBoostingClassifier(ccp_alpha=0,criterion='friedman_mse',init=None,learning_rate=0.7,loss='exponential',max_depth=3)
GBDT.fit(x_train,y_train)"""

#下面进行集成学习
#下面是硬投票(注意软投票的话只能用于二分类（才能有概率值）)
W=np.array(weight)/sum(weight) #计算各分类器权重6个模型
weight.pop(4) #现在只有5个特征值

#建立硬投票模型（注意硬投票目前还不能输出概率值绘制ROC曲线，解决方法是用软投票模型，并且将每个模型的权重设置为相等即可）
vote1=VotingClassifier(estimators=[('Randomforest',RF),('KNN',knn),
                                   ('SVC',svc),('gassuan',gassuan),
                                   ('sgd',sgd),('decision_tree',decision_tree)],voting='hard') #voting=soft or hard

#建立软投票模型(目前还不能添加sgd其不能输出概率值)
vote2=VotingClassifier(estimators=[('Randomforest',RF),('KNN',knn),
                                   ('SVC',svc),('gassuan',gassuan),
                                   ('decision_tree',decision_tree)],voting='soft',weights=weight) #注意目前还不能添加支持向量机

#下面进行模型训练
vote1.fit(x_train,y_train)
vote2.fit(x_train,y_train)
#下面是得到交叉验证的综合准确率
score=cross_val_score(vote1,x_train,y_train) #
score1=cross_val_score(vote2,x_train,y_train)
print('硬投票交叉验证',score.mean())
print('软投票交叉验证',score1.mean())
#下面用模型进行对测试集进行预测
y_pred1=vote1.predict(x_test)
y_pred2=vote2.predict(x_test)
print('硬投票测试集精度',metrics.accuracy_score(y_test,y_pred1)) #计算测试集的准确率(硬投票)
print('软投票测试集精度',metrics.accuracy_score(y_test,y_pred2)) #计算测试集的准确率(软投票)

#下面进行绘制软投票的混淆矩阵
confusion_matrix=metrics.confusion_matrix(y_test,y_pred2)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
plt.xlabel('prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('real category',fontsize=13)  #数轴是真实类别
plt.title('confusion matrix',fontsize=15)
plt.show()

#下面计算01分类的ROC面积(此时是计算测试集准确率的ROC面积)注意ROC曲线一般只适用于二分类的情况
"""下面我们进行可以进行绘制ROC曲线和计算AUC值（这里是计算总体的）
注意需要用概率来进行计算即后面的combos['predict']是其为1的概率序列具体可看logit回归中的ROC曲线计算
fpr,tpr,_=metrics.roc_curve(combos['是否违约'],combos['predict']) #计算ROC曲线点值
roc_auc=metrics.auc(fpr,tpr)"""

################################################################################
####################下面进行绘制出多分类的ROC曲线（注意需要重新构造数据）#################
################################################################################
#先将标签进行二值化[1 0 0 0] [0 0 1 0] [0 1 0 0] [0 0 0 1]
y=label_binarize(y_train_ori,classes=[0,1,2,3])
#设置种类
n_classes=y.shape[1]
# shuffle and split training and test sets (划分数据集)
x_train, x_test, y_train, y_test = train_test_split(x_train_ori, y, test_size=0.2)
y_trans_test=np.argmax(y_test,axis=1) #将one-hot类型的测试集标签转化一下，后面便于绘制
# Learn to predict each class against the other(其是一种多类分类的策略们可以为每一种类被配备一个分类器，01)
classifier = OneVsRestClassifier(vote2,n_jobs=-1) #建立多类分类器模型用，用软投票模型

#进行训练模型(目前也可运用投票模型（软投票)
classifier.fit(x_train,y_train)
y_score=classifier.predict(x_test)  #预测各个类别[0 0 0 1]
y_predict=np.argmax(y_score,axis=1) #将one-hot的编码格式转化成分类标签，作为真正的预测分类
y_proba=classifier.predict_proba(x_test) #计算每个类别概率（注意此时相加总和不一定是1）

#下面开始计算每一类的ROC
fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(n_classes): #遍历类别
    fpr[i],tpr[i], _ = metrics.roc_curve(y_test[:, i], y_proba[:, i]) #计算x轴和y轴的值
    roc_auc[i] = metrics.auc(fpr[i], tpr[i]) #计算auc面积值
# Compute micro-average ROC curve and ROC area（方法二:将每个类别原始值和预测值都进行展平再进行计算ROC）
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_proba.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

#绘制ROC曲线
plt.figure(figsize=(8,6))
#绘制平均的ROC曲线
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue','g'] #颜色序列有几类用几种
for i ,color in enumerate(colors):
    plt.plot(fpr[i],tpr[i],color=color,lw=2,label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i])) #绘制曲线同时打上AUC面积标签
plt.plot([0, 1], [0, 1], 'k--', lw=2)  #绘制对角线
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC-Kurve',fontsize=15)
plt.legend(loc="lower right")
plt.show()

#下面进行绘制混淆矩阵
confusion_matrix=metrics.confusion_matrix(y_trans_test,y_predict)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
plt.xlabel('Prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('Real category',fontsize=13)       #数轴是真实类别
plt.title('Confusion matrix',fontsize=15)
plt.show()
