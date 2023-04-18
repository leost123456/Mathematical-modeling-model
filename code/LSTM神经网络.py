import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras import optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

#下面是多维输入预测当前值的（一一对应的拟合）
#首先导入数据
data=pd.read_csv('D:\\数据\\bit_score1.csv')
data['earn_rate']=data['Value'].pct_change() #计算每日收益率(当前值减去前一个值再除去前一个值)
data['Date']=pd.to_datetime(data['Date'])
date_data=data['Date'] #存储下日期后面有用
data.fillna(0,inplace=True) #填补空缺
data.set_index(data['Date'],inplace=True) #将日期设置为索引，并且原地修改
data.drop(columns=['Date','score'],inplace=True,axis=1) #删去不需要的列

#下面进行数据标准化操作（0-1）
scaler=MinMaxScaler(feature_range=(-1,1))
scale_data=scaler.fit_transform(data.values[:,:-1])

#下面进行划分训练集和测试集（注意还可以设置一个最终的测试集，因为下面的这个是用于验证用的）
n=1500 #训练集截止序号
train_x,train_y=scale_data[:n,:-1],data.values[:n,-1] #训练集
test_x,test_y=scale_data[n:,:-1],data.values[n:,-1]

#下面进行转化成三维数据
train_X=train_x.reshape(train_x.shape[0],1,train_x.shape[1])
test_X=test_x.reshape(test_x.shape[0],1,test_x.shape[1])
train_Y=np.array(train_y) #y值坐标也要进行转换
test_Y=np.array(test_y)

#下面进行构建LSTM模型（Keras）
model=Sequential() #定义模型
model.add(LSTM(64,input_shape=(train_X.shape[1],train_X.shape[2]))) #注意结构（total_num,1,feature_num）
model.add(Dropout(0.5)) #加入0.5的Dropout
#添加输出层
model.add(Dense(units=1,input_dim=64,kernel_initializer='glorot_uniform'))
#Adam=optimizers.adam_v2.Adam(lr=0.01,decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None,amsgrad=False)
#下面进行按照验证集精度保存模型的最佳参数，其中monitor是指用什么参数来衡量,save_best_only表示只保存一种，mode是模式有（min,max,auto）
checkpoint = ModelCheckpoint('weights.hdf5' , monitor = 'val_loss' , save_best_only = True,mode='auto')
#下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.00001)
#下面进行配置模型参数
model.compile(loss='mse',optimizer='adam') #loss用mse来衡量
#下面进行训练模型操作(用测试集做验证)
history=model.fit(train_X,train_Y,batch_size=100,epochs=100,workers=0,validation_data=(test_X,test_Y), verbose=2,shuffle=False,callbacks=[checkpoint,reduce_lr]) #设置batch_size和epoch,同时还设置验证集为0.2,同时存储训练过程的各种数据给history

#下面进行绘制损失图
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize =13)     #坐标轴
plt.grid(alpha=0.3)                       #是否加网格线·
x=np.arange(len(history.history['loss'])) #设置x轴
plt.plot(x,history.history['loss'],label='Training Loss',color='b',alpha=0.7)
plt.plot(x,history.history['val_loss'],label='Valid Loss',color='r',alpha=0.7)
plt.legend()
plt.xlabel('epoch',fontsize=13)
plt.ylabel('loss',fontsize=13)
plt.title('Training Set And Test Set Loss',fontsize=15)
plt.show()

#下面进行预测验证集
y_valid=model.predict(test_X)

#下面进行可视化观察预测效果（盈利率）
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴(可以调节坐标轴字体来自适应标签，还可以自定义标签)
plt.grid(alpha=0.3)                    #是否加网格线·
x=np.arange(len(date_data[n:])) #设置x轴
plt.plot(date_data[n:],test_Y,label='original data',color='b',alpha=0.7)
plt.plot(date_data[n:],y_valid,label='predict data',color='r',alpha=0.7)
plt.legend()
plt.xticks(fontsize=13,rotation=30) #调整x轴的字体,进行30度旋转
plt.xlabel('Date',fontsize=13)
plt.ylabel('Earn Rate',fontsize=13)
plt.title('The Predicted Data Is Compared With The Original Data',fontsize=15)
plt.show()

#下面将盈利率乘进去得到股票的真实价值预测
val_value=[]
for step,i in enumerate(y_valid.ravel().tolist()):
    if step==0:
        val_value.append(data['Value'][n-2]*(1+i)) #注意这里要取前面的一个计算真实价值
    else:
        val_value.append(val_value[-1]*(1+i))
#val_value=pd.DataFrame(val_value).shift(periods=-1).values.tolist() #往前复原一个步长

#下面进行绘制真实价值预测和原始图对比
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴(可以调节坐标轴字体来自适应标签，还可以自定义标签)
plt.grid(alpha=0.3)                    #是否加网格线·
plt.plot(date_data[n:],data['Value'][1500:],label='original data',color='b',alpha=0.7)
plt.plot(date_data[n:],val_value,label='predict data',color='r',alpha=0.7)
plt.legend()
plt.xticks(fontsize=13,rotation=30) #调整x轴的字体,进行30度旋转
plt.xlabel('Date',fontsize=13)
plt.ylabel('Value',fontsize=13)
plt.title('The Predicted Data Is Compared With The Original Data',fontsize=15)
plt.show()

#下面继续模型评价（均方误差、均方根误差、R方、MAE）
#均方误差MSE
def mean_square_error(y1,y2): #y1是预测值序列，y2是真实值序列
    return np.sum((np.array(y1)-np.array(y2))**2)/len(y1)

#均方根误差（RMSE）
def root_mean_squard_error(y1,y2): #其中有y1是预测序列，y2是真实序列
    return np.sqrt(np.sum(np.square(np.array(y1)-np.array(y2)))/len(y1))

#平均绝对误差（MAE）
def mean_absolute_error(y1,y2):#其中y1是预测序列，y2是真实序列
    return np.sum(np.abs(np.array(y1)-np.array(y2)))/len(y1)

def computeCorrelation(x, y): #其中x,y均为序列，x是预测值，y是真实值,这里是计算Pearson相关系数，最后需要平方注意
    xBar = np.mean(x) #求预测值均值
    yBar = np.mean(y) #求真实值均值
    covXY = 0
    varX = 0          #计算x的方差和
    varY = 0          #计算y的方差和
    for i in range(0, len(x)):
        diffxx = x[i] - xBar  #预测值减预测值均值
        diffyy = y[i] - yBar  #真实值减真实值均值
        covXY += (diffxx * diffyy)
        varX += diffxx ** 2
        varY += diffyy ** 2
    return covXY/np.sqrt(varX*varY)

#下面进行输出各个指标数值
print(f'MSE:{mean_square_error(y_valid,test_Y)}\n'
      f'RMSE:{root_mean_squard_error(y_valid,test_Y)}\n'
      f'MAE:{mean_absolute_error(y_valid,test_Y)}\n'
      f'R方:{computeCorrelation(y_valid,test_Y)**2}\n')

##################################################
#下面进行构造多个维度的输入且是多个时间步长
##################################################
#下面是超参数
n_feature=6 #特征数量
n_past=30 #，创建数据集的时候，设置步长为30，也就是用前面30组数据预测下一个值（在）

#下面进行数据缩放(注意也将目标值也进行缩放操作了) 变成shape为（w,h）形式
value=data['Value'].tolist()
data.drop('Value',axis=1,inplace=True)
data['Value']=value #将其变成最后一列
scaler=MinMaxScaler(feature_range=(0,1))
data_scale=scaler.fit_transform(data.values)

#下面进行拆分数据（X和Y，设置步长预测下一个值）注意这里是针对多维数据的，仅仅就是时间序列的预测还要单独搞一下
def Resetting_data(data,n_past): #其中data是输入的数据集（矩阵形式(二维)）,n_past表示（步长）用之前的几个数据预测后一个值
    data_X=[] #存储步长为n_past数量的数据
    data_Y=[] #预测的下一个目标值，注意最后一个值就是对应着最后一个
    for i in range(n_past,len(data)):
        data_X.append(data[i-n_past:i,:-1])
        data_Y.append(data[i,-1])
    return np.array(data_X),np.array(data_Y) #最后的shape为（n,w,h）三维

new_x,new_y=Resetting_data(data_scale,n_past) #对所有数据集拆分，设置步长为30,返回的是三维数据

test_split=round(0.2*data.values.shape[0]) #测试集的占比
train_x=new_x[:-test_split,:,:]
test_x=new_x[-test_split:,:,:]
train_y=new_y[:-test_split]
test_y=new_y[-test_split:]

#下面利用网格搜索的思想寻找最优的batch_size和epoch还有优化器(还未成功)
'''def find_bestmodel(optimizer): #ootimizer是优化器种类
    grid_model=Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(n_past,train_x.shape[2])))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    #添加输出层
    grid_model.add(Dense(units=1,input_dim=50,kernel_initializer='glorot_uniform')) #kernel_initializer='glorot_uniform'
    grid_model.compile(loss='mae',optimizer=optimizer) #模型配置

grid_model = KerasRegressor(build_fn=find_bestmodel, verbose=1, validation_data=(test_x, test_y)) #创建一个训练器
parameters = {'batch_size' : [8,32],
              'epochs' : [8,40],
              'optimizer' : ['adam','Adadelta'] }
grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 3) #网格搜索加交叉验证（三折）
grid_search=grid_search.fit(train_x,train_y) #进行训练'''

#下面进行训练数据（可以进行各种调参（利用网格搜索的方式），不过下面就是手动调参）双层LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(n_past,train_x.shape[2]))) #隐含层为50个神经元,returen sequences是返回最后的结果
model.add(LSTM(50))
model.add(Dropout(0.2))
#添加输出层
model.add(Dense(units=1,input_dim=50,kernel_initializer='glorot_uniform')) #kernel_initializer='glorot_uniform'

#下面进行按照验证集精度保存模型的最佳参数，其中monitor是指用什么参数来衡量,save_best_only表示只保存一种，mode是模式有（min,max,auto）
checkpoint = ModelCheckpoint('weights.hdf5' , monitor = 'val_loss' , save_best_only = True,mode='auto')
#下面是微调学习率，如果在5轮内验证集精度未上升，则学习率减半
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, min_lr=0.00001)
#下面进行配置模型参数
SGD=optimizers.gradient_descent_v2.SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mae',optimizer='adam') #loss用mse来衡量
#下面进行训练模型操作(用测试集做验证)
history=model.fit(train_x,train_y,batch_size=8,epochs=30,workers=-1,validation_data=(test_x,test_y), verbose=2,shuffle=False,callbacks=[checkpoint,reduce_lr]) #设置batch_size和epoch,同时还设置验证集为0.2,同时存储训练过程的各种数据给history

#下面进行绘制损失图
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize =13)     #坐标轴
plt.grid(alpha=0.3)                       #是否加网格线·
x=np.arange(len(history.history['loss'])) #设置x轴
plt.plot(x,history.history['loss'],label='Training Loss',color='b',alpha=0.7)
plt.plot(x,history.history['val_loss'],label='Valid Loss',color='r',alpha=0.7)
plt.legend()
plt.xlabel('epoch',fontsize=13)
plt.ylabel('loss',fontsize=13)
plt.title('Training Set And Test Set Loss',fontsize=15)
plt.show()

#下面预测测试集同时将结果进行还原（因为原来对目标值做了标准化）
model.load_weights('weights.hdf5') #加载最佳的模型
predict_y=model.predict(test_x) #得到预测值
predict_y_copy=np.repeat(predict_y,n_feature+1,axis=-1) #表示对predict_y进行数据增广操作，竖向复制成n_feature+1维(所有特征加上一列的目标值列)，axis=-1表示新增维度进行增广（这里是指1按列）
predict_y=scaler.inverse_transform(np.reshape(predict_y_copy,(len(test_y),n_feature+1)))[:,-1] #进行特征逆变换,同时只需要取其中一列就可以了,最终输出

#下面进行可视化最终结果操作（注意需要把原始的值往后移一个步长，因为我们预测出的结果是针对下一个的）
data['Value']=data['Value'].shift(periods=1)      #针对于原始值，往后移动一个单位
data['Value'].fillna(method='bfill',inplace=True) #用后一个不是缺失值的数据进行代替缺失值
new_test_y=data['Value'][-test_split:]
#下面进行绘制真实价值预测和原始图对比
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴(可以调节坐标轴字体来自适应标签，还可以自定义标签)
plt.grid(alpha=0.3)                    #是否加网格线·
plt.plot(date_data[-test_split:],new_test_y,label='original data',color='b',alpha=0.7)
plt.plot(date_data[-test_split:],predict_y,label='predict data',color='r',alpha=0.7)
plt.legend()
plt.xticks(fontsize=13,rotation=30) #调整x轴的字体,进行30度旋转
plt.xlabel('Date',fontsize=13)
plt.ylabel('Value',fontsize=13)
plt.title('The Predicted Data Is Compared With The Original Data',fontsize=15)
plt.show()

#下面就是输出评价指标(针对原始值的)
print('MSE:',mean_square_error(predict_y,new_test_y))
print('RMSE',root_mean_squard_error(predict_y,new_test_y))
print('MAE',mean_absolute_error(predict_y,new_test_y))
print('R方',computeCorrelation(predict_y,new_test_y)**2)

#再下面就是进行后面未知的(用迭代的思想，用最优模型预测一个步长，再回测特征再送进模型预测)