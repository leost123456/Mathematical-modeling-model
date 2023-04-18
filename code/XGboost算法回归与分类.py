import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams #导入包
from sklearn import metrics
import seaborn as sns
from sklearn.datasets import load_boston #导入房价数据

#改变字体
config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置
#首先导入鸢尾花数据集（分类数据）四维特征
iris=load_iris()
x=np.array(iris.data) #特征集
y=np.array(iris.target) #标签
#下面进行划分训练集和测试集
x_test=x[-6:,:]
y_test=y[-6:]
#下面进行划分训练集和验证集
x_train, x_valid, y_train, y_valid = train_test_split(x[:-6,:], y[:-6], test_size=0.2)
#下面进行重构成xgboost的输入数据（训练集、验证机和测试集都要）
train=xgb.DMatrix(x_train,label=y_train)
valid=xgb.DMatrix(x_valid,label=y_valid)
test=xgb.DMatrix(x_test) #测试集转换可以不用输入标签
# 参数设置
params = {
    'booster': 'gbtree', #基分类器的类型 可以是ugbtree、gblinear和dart
    'objective':'multi:softmax', #目标的类型（具体可以查看文本），多分类的就是multi:softmax 回归就选reg:linear 二分类就选binary:logistic
    'num_class': 3, #标签的分类类别数
    'max_depth':6,  #最大树深，越大越容易过拟合。[1,+∞]之间
    'eta':0.3,#即learning_rate。[0,1]之间。
    'silent':1, #0状态会打印信息，1表示不打印信息
    'gamma':0, #在树的叶子节点上进一步划分的最小损失量，值越大，算法就越保守。[0,+∞]之间。
    'min_child_weight':1,#树构建过程中，当节点的权重小于这个值则分支会被丢弃，值越大，算法越保守。[0,+∞]之间。
    'subsample':1, #样本的采样比例，值越大，越容易过拟合。(0,1]之间。
    'colsample_bytree':1, #构建每棵树时列的采样比率。(0,1]之间。
    'colsample_bylevel':1, #每一级的每一次分裂的采样比率。(0,1]之间。
    'lambda':1,#L2正则。
    'alpha':0, #L1正则。
    'nthread':-1,
    'eval_metric':'merror', #评估方法，分类就用merror(错误率)、logloss（交叉熵损失函数），，回归可以用rmse,mae,二分类可以用auc
    'seed':0,
}
#下面进行模型训练(用训练集和验证集) 其中num_boost_round是迭代的次数，也可以称为最终迭代后树的数量，每次增加迭代增加一个弱分类器最终达到
model=xgb.train(params,train,num_boost_round=100, evals=[(valid,'eval')]) #注意训练过程中会不断输出每一轮的验证集评估方法merror

#下面是绘制特征重要性条形图
#首先获取各特征的重要性并进行升序排序
sort_list=[(name,value) for name,value in model.get_fscore().items()] #注意刚开始的顺序就是从第一列特征f0开始的（也就是后面的name_list可以更换成各特征原始名称）
sort_list=sorted(sort_list,reverse=False,key=lambda x:x[1]) #降序
name_list=[x[0] for x in sort_list]  #存储排序后的特征名称列表
value_list=[x[1] for x in sort_list] #存储排序后的特征Fscore值
#下面进行绘制竖着的条形图F score（注意是从下往上绘制的，因此原来的序列是升序的）
height=0.25 #高度（每个条形之间）
index=np.arange(x_train.shape[1])      #特征类别个数
plt.figure(figsize=(7,5))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
plt.barh(index,value_list,height=height,color='#F21855',alpha=1)
#下面进行打上标签(注意细节，交换了y/x)
for i,data in enumerate(value_list):
    plt.text(data+7,index[i]-0.06,round(data,1),horizontalalignment='center',fontsize=13) #具体位置可以进行微调
plt.xlim(0,max(value_list)+30) #改变x轴的显示范围
plt.yticks(index,name_list)    #设置y轴的标签
plt.xlabel('F score',fontsize=13)  #注意这个还是横轴的
plt.ylabel('Features',fontsize=13) #竖轴的标签
plt.title('Feature importance',fontsize=15)
plt.show()

#下面进行测试集的预测
pred=model.predict(test)

#下面进行计算测试集的精度
acc=metrics.accuracy_score(y_test,pred)
print(acc)

#下面进行绘制混淆矩阵
confusion_matrix=metrics.confusion_matrix(y_test,pred)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
plt.xlabel('prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('real category',fontsize=13)       #数轴是真实类别
plt.title('confusion matrix',fontsize=15)
plt.show()

#对于分类模型来说（注意还可以输出概率，得到ROC曲线）

#敏感性分析可以通过改变树的深度（max_depth）、弱分类器的数量(num_round_boost)、迭代次数、输入特征的维度(随机)、损失的选择等等
#注意如果想对模型进行交叉验证的话，用下列的方式
new_model=XGBClassifier(n_estimators=100) #其中n_estimators等价于num_boost_bound
score=cross_val_score(new_model,x[:-6,:],y[:-6],cv=5) #进行五折交叉验证
print(score.mean()) #输出

#下面进行XGboost回归（波士顿房价数据）十三维的特征
boston = load_boston()
x1=np.array(boston.data)
y1=np.array(boston.target)
#划分测试集
x_test1=x1[-50:,:]
y_test1=y1[-50:]
#下面进行划分训练集和验证集
x_train1, x_valid1, y_train1, y_valid1 = train_test_split(x1[:-50,:], y1[:-50], test_size=0.2)
#转变数据格式
train1 = xgb.DMatrix(x_train1, label=y_train1)
valid1 = xgb.DMatrix(x_valid1, label=y_valid1)
test1 = xgb.DMatrix(x_test1)
total_data1=xgb.DMatrix(x1[:-50,:]) #包含训练集和验证集的数据用于后续的绘制拟合图像
total_data2=xgb.DMatrix(x1) #这里是包含所有数据的，用于后续的模型评价分析
# 参数设置
params1 = {
    'booster': 'gbtree',
    'objective':'reg:linear',
    'max_depth':6,
    'eta':0.1,
    'silent':1,
    'gamma':0,
    'min_child_weight':1,
    'subsample':1,
    'colsample_bytree':1,
    'colsample_bylevel':1,
    'lambda':1,
    'alpha':0,
    'nthread':-1,
    'eval_metric':'rmse',
    'seed':0
}
# 模型训练
model1 = xgb.train(params1,train1,num_boost_round=100,evals=[(valid1,'eval')]) #其中num_boost_round是迭代的次数也可以理解为树的数量

#下面是绘制特征重要性条形图
#首先获取各特征的重要性并进行升序排序
sort_list1=[(name,value) for name,value in model1.get_fscore().items()] #注意刚开始的顺序就是从第一列特征f0开始的（也就是后面的name_list可以更换成各特征原始名称）
sort_list1=sorted(sort_list1,reverse=False,key=lambda x:x[1]) #降序
name_list1=[x[0] for x in sort_list1] #存储排序后的特征名称列表
value_list1=[x[1] for x in sort_list1] #存储排序后的特征Fscore值

#下面进行绘制竖着的条形图F score（注意是从下往上绘制的，因此原来的序列是升序的）
height=0.25 #高度（每个条形之间）
index1=np.arange(x_train1.shape[1])      #特征类别个数
plt.figure(figsize=(7,5))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
plt.barh(index1,value_list1,height=height,color='#F21855',alpha=1)
#下面进行打上标签(注意细节，交换了y/x)
for i,data in enumerate(value_list1):
    plt.text(data+29,index1[i]-0.16,round(data,1),horizontalalignment='center',fontsize=13) #具体位置可以进行微调
plt.xlim(0,max(value_list1)+65) #改变x轴的显示范围
plt.yticks(index1,name_list1) #设置y轴的标签
plt.xlabel('F score',fontsize=13) #注意这个还是横轴的
plt.ylabel('Features',fontsize=13) #竖轴的标签
plt.title('Feature importance',fontsize=15)
plt.show()

#下面还可以进行绘制模型损失函数在每一轮迭代的变化情况图(将打印的信息全部整合到一个csv文件中再进行绘制即可)
pred_train_valid=model1.predict(total_data1).tolist() #训练集预测
pred_test=model1.predict(test1).tolist() #测试集预测
pred_test1=[pred_train_valid[-1]]+pred_test  #注意绘制预测的曲线时要在列表最前面加上train和valid中最后一个数

#下面进行绘制的拟合效果和测试效果（训练集、验证集和测试集）
plt.figure(figsize=(7,5))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
#先绘制原始数据(注意x轴的序号)
plt.scatter(np.arange(1,len(y1[:-50])+1),y1[:-50],marker='o',color='#2049ff',s=15,label='Raw data for training set and validation set') #训练集和验证集
plt.scatter(np.arange(len(y1[:-50])+1,len(y1)+1),y1[-50:],marker='^',color='#FFDD24',s=15,label='Test set raw data') #测试集
#下面进行绘制预测的曲线(注意第二条线x轴的序号，要前移一位)
plt.plot(np.arange(1,len(y1[:-50])+1),pred_train_valid,color='#52D896',lw=1,ls='solid',label='Prediction data for training set and test set')
plt.plot(np.arange(len(y1[:-50]),len(y1)+1),pred_test1,color='#F21855',lw=1,ls='solid',label='Test set prediction data')
#下面进行绘制分割线
plt.axvline(x=len(y1[:-50]), color='#bf0000',lw=2,linestyle='dashed')
#下面进行打上文字标签
plt.text(480,45,s='Test set',fontsize=10)
plt.text(-12,48,s='Training set and validation set',fontsize=10)
#下面进行其余的细节设置
plt.legend()
plt.xlabel('Index',fontsize=13)
plt.ylabel('Price',fontsize=13)
plt.title('Regression prediction of house prices',fontsize=15)
plt.show()

#下面可以对所有数据进行R方、MSE等指标的运算（注意还可以输出概率，得到ROC曲线）
pred_total=model1.predict(total_data2) #得到所有预测的数据

#第二种计算R方的方式 (1-(SSE/SST))
def R_Squre2(y_pred,y_real): #其中y_pred是预测值，y_real是真实值,两者都是序列
    y_real_mean = np.mean(y_real)
    y_pred_var=0 #计算预测值减真实值的平方和
    y_real_var=0
    for i in range(len(y_pred)):
        y_pred_var+=(y_pred[i]-y_real[i])**2
        y_real_var+=(y_real[i]-y_real_mean)**2
    return 1-y_pred_var/y_real_var

print(R_Squre2(pred_total,y1)) #R方值
