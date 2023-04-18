import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import tensorflow as tf
from keras.utils.vis_utils import  plot_model,model_to_dot #绘制模型配置图
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn import metrics

#先进行多维输入和单维输出的神经网络（分类、回归均可）
pre_data=pd.DataFrame(pd.read_excel('D:\\国赛数模优秀论文\\2017\\Problems\\B\\附件一：已结束项目任务数据.xls')) #原始训练数据
new_data=pd.DataFrame(pd.read_excel('D:\\国赛数模优秀论文\\2017\\Problems\\B\\附件三：新项目任务数据.xls'))
pre_data.drop(columns='任务号码',inplace=True)
new_data.drop(columns='任务号码',inplace=True)
X=pre_data.iloc[:,:2] #原始X数据
Y=pre_data.iloc[:,-1] #原始标签数据

#下面进行随机划分数据集（还可以用交叉验证进行更好的改进）
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,shuffle=True) #随机划分训练集和测试集,一定要打乱，能提高精度

#下面进行数据的标准化（均值标准差的那个）
sc=StandardScaler() #创建归一器
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#定义随机种子（使其能够复现）
#np.random.seed(7)
#先初始化一个BP神经网络
model=Sequential()
#加入输入层
model.add(Dense(units=100,input_dim=2,kernel_initializer='glorot_uniform',activation='relu')) #unit是神经元的数量,input_dim是输入的纬度，activation是激活函数
model.add(Dropout(rate=0.1)) #加入dropout防止过拟合
#加入隐含层
model.add(Dense(units=50,input_dim=100,kernel_initializer='glorot_uniform',activation='relu')) #这里的input_dim要与上一层的神经员个数对上
model.add(Dropout(rate=0.1)) #加入dropout防止过拟合
model.add(Dense(units=50,input_dim=50,kernel_initializer='glorot_uniform',activation='relu'))
model.add(Dropout(rate=0.1)) #加入dropout防止过拟合
model.add(Dense(units=20,input_dim=50,kernel_initializer='glorot_uniform',activation='relu'))
model.add(Dropout(rate=0.1)) #加入dropout防止过拟合
model.add(Dense(units=10,input_dim=20,kernel_initializer='glorot_uniform',activation='relu'))
model.add(Dropout(rate=0.1)) #加入dropout防止过拟合
#添加输出层
model.add(Dense(units=1,input_dim=10,kernel_initializer='glorot_uniform'))
model.add(Activation('sigmoid')) #也可以这么写 在0-1分类的时候最后用sigmoid、

#下面进行优化器的配置(自定义优化器可以进行微调操作)
SGD=optimizers.gradient_descent_v2.SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
Adam=optimizers.adam_v2.Adam(lr=0.01,decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None,amsgrad=False)

#下面进行按照验证集精度保存模型的最佳参数，其中monitor是指用什么参数来衡量,save_best_only表示只保存一种，mode是模式有（min,max,auto）
checkpoint = ModelCheckpoint('weights.hdf5' , monitor = 'val_accuracy' , save_best_only = True,mode='auto') #注意monitor目前这个设置的是分类问题的，回归问题要换成MSE

#下面进行配置模型的提早退出，如果验证集精度在5轮没有上升则退出训练。
#early_stopping = EarlyStopping(monitor = 'val_accuracy' , patience = 5)

#下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.0001)

#下面进行配置模型参数
model.compile(optimizer=SGD,loss='binary_crossentropy',metrics=['accuracy']) #这里用交叉熵损失函数，注意可以进行转换，回归问题可以用mean_square_error,也就是说模型的loss和精度都用MSE表示,metrix回归问题用’mse‘

#可视化模型
plot_model(model,to_file='model.png',show_shapes=True,dpi=96) #绘制模型的配置图

#下面进行训练操作
history=model.fit(x_train,y_train,batch_size=8,epochs=50,workers=4,validation_split=0.2,callbacks=[checkpoint,reduce_lr]) #设置batch_size和epoch,同时还设置验证集为0.2,同时存储训练过程的各种数据给history

#读取训练过程中的参数
train_loss=history.history['loss'] #寻来你及
val_loss=history.history['val_loss']
train_accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']
epochs=np.arange(1,len(train_accuracy)+1)    #轮数序列

#下面进行绘制训练精度和验证集精度曲线，还有训练集的loss和验证集的loss曲线
plt.figure(figsize=(12,6)) #长、宽
plt.subplot(121)
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.plot(epochs,train_loss,marker='o',lw=2,label='train loss')
plt.plot(epochs,val_loss,marker='*',lw=2,label='val loss')
plt.xlabel('epochs',fontsize=13)
plt.ylabel('loss',fontsize=13)
plt.title('Training set and test set loss change',fontsize=15)
plt.legend()
plt.subplot(122)
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.plot(epochs,train_accuracy,marker='o',lw=2,label='train loss')
plt.plot(epochs,val_accuracy,marker='*',lw=2,label='val loss')
plt.xlabel('epochs',fontsize=13)
plt.ylabel('accuracy',fontsize=13)
plt.title('Training set and test set accuracy change',fontsize=15)
plt.legend()
plt.show()

#下面进行预测
model.load_weights('weights.hdf5') #载入最佳参数的模型
y_pred_prob=model.predict(x_test) #概率预测，可以用于绘制ROC曲线
y_pred=[] #存储预测值
for data in y_pred_prob:
    if data>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

fpr,tpr,_=metrics.roc_curve(y_test,y_pred_prob) #计算ROC曲线点值
roc_auc=metrics.auc(fpr,tpr) #计算ROC曲线下的面积

#下面进行绘制测试集的ROC曲线（前面都是训练集和验证集）注意如果是多分类的话，就对原始数据分类进行one-hot编码，同时最后一层利用sigmoid函数
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

#下面进行绘制测试集的混淆矩阵
confusion_matrix=metrics.confusion_matrix(y_test,y_pred) #求出混淆矩阵
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
plt.xlabel('prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('real category',fontsize=13)  #数轴是真实类别
plt.title('confusion matrix',fontsize=15)
plt.show()


###########################################################################################
########################################交叉验证使用#########################################
###########################################################################################

#下面进行交叉验证（思路是不将训练集进行分离了，在训练过程中用keras中自带的validation进行每次不断的测试，而把交叉验证分出来验证集作为最终的测试集进行使用）
#先进行多维输入和单维输出的神经网络（分类、回归均可）
pre_data=pd.DataFrame(pd.read_excel('D:\\国赛数模优秀论文\\2017\\Problems\\B\\附件一：已结束项目任务数据.xls')) #原始训练数据
new_data=pd.DataFrame(pd.read_excel('D:\\国赛数模优秀论文\\2017\\Problems\\B\\附件三：新项目任务数据.xls'))
pre_data.drop(columns='任务号码',inplace=True)
new_data.drop(columns='任务号码',inplace=True)
X=pre_data.iloc[:,:2] #原始X数据
Y=pre_data.iloc[:,-1] #原始标签数据
#下面进行数据的标准化（均值标准差的那个,且作用于dataframe，同时还将其转化未nddarry格式）
sc=StandardScaler()    #创建归一器
X=sc.fit_transform(X)

#使用分层交叉验证（每一折中每个标签的类别数相同）同时打乱原始的数据集先
kf=StratifiedKFold(n_splits=5, shuffle=True) #5折交叉验证
score_list=[] #存储每次交叉验证分出的测试集的精确度，注意这里将数据转化成了ndarray的数据格式，使其能够在下面用索引取值

for train_index,test_index in kf.split(X,Y):
    # 先初始化一个BP神经网络
    model = Sequential()
    # 加入输入层
    model.add(Dense(units=100, input_dim=2, kernel_initializer='glorot_uniform',
                    activation='relu'))  # unit是神经元的数量,input_dim是输入的纬度，activation是激活函数
    model.add(Dropout(rate=0.1))  # 加入dropout防止过拟合
    # 加入隐含层
    model.add(Dense(units=50, input_dim=100, kernel_initializer='glorot_uniform',
                    activation='relu'))  # 这里的input_dim要与上一层的神经员个数对上
    model.add(Dropout(rate=0.1))  # 加入dropout防止过拟合
    model.add(Dense(units=50, input_dim=50, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(rate=0.1))  # 加入dropout防止过拟合
    model.add(Dense(units=20, input_dim=50, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(rate=0.1))  # 加入dropout防止过拟合
    model.add(Dense(units=10, input_dim=20, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(rate=0.1))  # 加入dropout防止过拟合
    # 添加输出层
    model.add(Dense(units=1, input_dim=10, kernel_initializer='glorot_uniform'))
    model.add(Activation('sigmoid'))  # 也可以这么写 在0-1分类的时候最后用sigmoid、
    # 下面进行优化器的配置(自定义优化器可以进行微调操作)
    SGD = optimizers.gradient_descent_v2.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Adam = optimizers.adam_v2.Adam(lr=0.01, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    # 下面进行按照验证集精度保存模型的最佳参数，其中monitor是指用什么参数来衡量,save_best_only表示只保存一种，mode是模式有（min,max,auto）
    checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_accuracy', save_best_only=True, mode='auto')
    # 下面进行配置模型的提早退出，如果验证集精度在5轮没有上升则退出训练。
    # early_stopping = EarlyStopping(monitor = 'val_accuracy' , patience = 5)
    # 下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
    # 下面进行配置模型参数
    model.compile(optimizer=SGD, loss='binary_crossentropy', metrics=['accuracy'])  # 这里用交叉熵损失函数，注意可以进行转换
    history=model.fit(X[train_index],Y[train_index],batch_size=8,epochs=50,workers=4,validation_split=0.2,callbacks=[checkpoint,reduce_lr]) #设置batch_size和epoch,同时还设置验证集为0.2,同时存储训练过程的各种数据给history

    #下面开始计算每一次交叉验证的测试集精确度
    score = model.evaluate(X[test_index], Y[test_index], verbose=0)
    score_list.append(score) #存储到列表中

    #读取训练过程中的参数
    train_loss=history.history['loss'] #寻来你及
    val_loss=history.history['val_loss']
    train_accuracy=history.history['accuracy']
    val_accuracy=history.history['val_accuracy']
    epochs=np.arange(1,len(train_accuracy)+1)    #轮数序列
    #下面如果想画图的话就和上面一样的操作复制即可

print(np.mean(score_list)) #输出交叉验证综合精度


