import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from matplotlib import rcParams #导入包

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#导入数据，并提取出需要用的数据
data=pd.read_csv('D:\\2023美赛\\2023\\处理后的数据.csv')[['Date','Number of reported results']]
#转化时间格式
data['Date']=pd.to_datetime(data['Date'])
data.rename(columns={'Date':'ds','Number of reported results':'y'},inplace=True)

#下面进行构Prophet模型
#注意
model = Prophet(growth="linear", #趋势项模型，还可以设置为logistic,注意设置为logistic时，要确保原数据有一列数据为承载力上限（要自己设置），例如data['cap']=3.5
                #下面是用于趋势项变点的设置参数，chagepoints表示人为设置变点，传入参数为时间的列表,n_changepoints表示变点的数量，chagepoint_prior_scale表示变点的强度影响
                changepoints=None,n_changepoints=25,changepoint_prior_scale=0.05,
                #下面是进行添加节假日信息,holidays为传入的节假日时间列表，holidays_prior_scale表示节假日影响因子。
                holidays=None,holidays_prior_scale=10.0,
                interval_width = 0.95,   #获取95%的置信区间
                )

#下面可以进行自定义周期
#其中name是名称，可以任意取，period是周期，fourier_order是级数的数量（根据经验一般7天选3，一年选择10），prior_scale就是季节性强度参数可以用于后续的敏感性分析
model.add_seasonality(name='weekly',period=7,fourier_order=3,prior_scale=10.0)

#下面进行添加节假日信息
model.add_country_holidays('CN') #可以添加某个国家的节假日信息,如US、CN等等

#下面进行拟合模型
model.fit(data)

#下面进行设置预测的时间列(预测往后的30天)(注意其同时也包括了原始的时间列)
period=30
future=model.make_future_dataframe(periods=period,freq='D')

#下面进行最终模型的预测
forecast=model.predict(future) #注意其中包含了很多列数据，主要需要的就是yhat/yhat_lower/yhat_uuper/trend/weekly（自己前面设置的周期性因子）/holidays(前面设置的节假日因子)
change_points=model.changepoints #获取变点的x轴坐标（时间）

#下面进行绘制拟合和预测的曲线（包含置信区间）
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
#先绘制原始数据点
plt.scatter(data['ds'],data['y'],marker='o',color='#2049ff',s=20,label='Raw data')
#下面绘制预测的曲线
plt.plot(future,forecast['yhat'],color='#F21855',lw=2,label='Forecast curve')
#下面进行绘制95%置信区间
plt.fill_between(future['ds'].tolist(),forecast['yhat_lower'],forecast['yhat_upper'],color='#F21855',alpha=0.3,label='95% confidence interval')
#划分训练集和预测集
plt.axvline(x=future.iloc[-period], color='#bf0000',lw=2,linestyle=':')
plt.text(future.iloc[-period-30],300000,s='Past',fontsize=15)
plt.text(future.iloc[-period+5],300000,s='Future',fontsize=15)
plt.legend()
plt.title('Model prediction results',fontsize=15)
plt.xlabel('Fate',fontsize=13)
plt.ylabel('Value',fontsize=13)
plt.show()

#下面进行时间序列分解
fig,ax=plt.subplots(3,1,figsize=(8,6))
#trend
ax[0].plot(future,forecast['trend'],label='Trend',color='#F21855',lw=1.5,alpha=1)
#下面进行绘制趋势因子的上下限
ax[0].fill_between(future['ds'].tolist(),forecast['trend_lower'],forecast['trend_upper'],alpha=0.3,color='#F21855',label='95% confidence interval')
#下面添加箭头表示趋势（可以根据实际情况操作）
#ax[0].annotate('', xy=(trend.index[-15], 25), xytext=(trend.index[-40],30),
             #arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2,alpha=0.6))
print(forecast['weekly'])
ax[0].legend()
ax[0].set_ylabel('Trend',fontsize=13)
ax[0].set_title('Time series decomposition results',fontsize=15)
ax[0].legend()
ax[0].tick_params(size=5,labelsize = 13) #坐标轴
ax[0].grid(alpha=0.3)                    #是否加网格线
#holidays
ax[1].plot(future,forecast['holidays'],label='Holidays',color='#52D896',lw=1.5,alpha=1)
ax[1].set_ylabel('Holidays',fontsize=13)
ax[1].legend()
ax[1].tick_params(size=5,labelsize = 13) #坐标轴
ax[1].grid(alpha=0.3)                    #是否加网格线
#seasonal（根据自己设置的周期）注意还可以只绘制出一个周期来表示
ax[2].plot(future,forecast['weekly'],label='Seasonal',color='#FFDD24',alpha=1)
ax[2].set_xlabel('Date',fontsize=13)
ax[2].set_ylabel('Seasonal',fontsize=13)
ax[2].legend()
ax[2].tick_params(size=5,labelsize = 13) #坐标轴
ax[2].grid(alpha=0.3)                    #是否加网格线"""
plt.tight_layout() #自动调整子图间距
plt.show()

#下面进行验证模型的验证精度，可以用均方误差、R方，aic、bic，前面分成训练集和验证集来进行等等
#均方误差MSE
def mean_square_error(y1,y2): #y1是预测值序列，y2是真实值序列
    return np.sum((np.array(y1)-np.array(y2))**2)/len(y1)

#第二种计算R方的方式 (1-(SSE/SST))
def R_Squre2(y_pred,y_real): #其中y_pred是预测值，y_real是真实值,两者都是序列
    y_real_mean = np.mean(y_real)
    y_pred_var=0 #计算预测值减真实值的平方和
    y_real_var=0
    for i in range(len(y_pred)):
        y_pred_var+=(y_pred[i]-y_real[i])**2
        y_real_var+=(y_real[i]-y_real_mean)**2
    return 1-y_pred_var/y_real_var

print('R方为{}'.format(R_Squre2(forecast['yhat'][:-period],data['y'])))
print('均方误差为{}'.format(mean_square_error(forecast['yhat'][:-period],data['y'])))
print(data['ds'])
#下面进行模型的敏感性分析，主要对三个参数changepoint_prior_scale/holidays_prior_scale/seasonal_prior_scale进行分析
#进行趋势项因子强度的敏感性分析(变点强度)
trend_list=[0.1,0.3,0.5,0.7,0.9]
result1=[] #用于存储每一次预测的值形式为[[],[]]
for i in trend_list:
    model1 = Prophet(growth="linear", #趋势项模型，还可以设置为logistic,注意设置为logistic时，要确保原数据有一列数据为承载力上限（要自己设置），例如data['cap']=3.5
                    #下面是用于趋势项变点的设置参数，chagepoints表示人为设置变点，传入参数为时间的列表,n_changepoints表示变点的数量，chagepoint_prior_scale表示变点的强度影响
                    changepoints=None,n_changepoints=25,changepoint_prior_scale=i,
                    #下面是进行添加节假日信息,holidays为传入的节假日时间列表，holidays_prior_scale表示节假日影响因子。
                    holidays=None,holidays_prior_scale=10.0,
                    interval_width = 0.95,   #获取95%的置信区间
                    )
    #下面可以进行自定义周期
    #其中name是名称，可以任意取，period是周期，fourier_order是级数的数量（根据经验一般7天选3，一年选择10），prior_scale就是季节性强度参数可以用于后续的敏感性分析
    model1.add_seasonality(name='weekly',period=7,fourier_order=3,prior_scale=10.0)
    model1.fit(data)
    #下面进行预测并存储数据，注意输入预测的形式的列名一定要有且是'ds'才行
    result1.append(model1.predict(pd.DataFrame({'ds':data['ds'].tolist()}))['yhat'].tolist())#只存储预测的值，其余不管
#输出
print(result1)
#下面进行季节性强度因子和节假日强度因子的敏感性分析同理
#注意可以将三个敏感性分析绘制成三个横向的折线图形式（还可以分别加上阴影和箭头标签）
