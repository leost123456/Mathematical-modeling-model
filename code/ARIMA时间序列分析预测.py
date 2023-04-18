import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.api import qqplot
import itertools
import seaborn as sns
from scipy.stats import t

#主要的思想就是先利用训练数据进行定阶数（其实也就是分成训练集测试集的思想），对训练数据的模型进行检验，最后将全部数据送入模型对后面未知的数据进行预测

warnings.filterwarnings('ignore')
#首先导入数据
data=pd.read_csv('D:\\学习资源\\学习资料\\数据\\ChinaBank.csv') #将Date列设置为索引,parse_dates是将日期格式进行转化
#或者也可以利用下面的代码进行日期转化
data['Date']=pd.to_datetime(data['Date'])
data=data[['Date','Close']]
total_data=data.iloc[:127,:] #读取需要的数据

#下面进行展示序列(这里是展示选取的)
'''plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线·
plt.plot(train_data['Date'],total_data['Close'])
plt.show()'''

#下面进行平稳性检验，首先是差分方法同时展示图(这个可以进行主观的观察下数据)
total_data['Close_diff1']=total_data['Close'].diff(1) #一阶差分
total_data['Close_diff2']=total_data['Close_diff1'].diff(1) #二阶差分
total_data['Close_diff1']=total_data['Close_diff1'].fillna(0) #填补空值
total_data['Close_diff2']=total_data['Close_diff2'].fillna(0) #填补空值
print(total_data['Close_diff1'])
print(total_data['Close_diff2'])
train_data=total_data.iloc[:84,:] #划分训练集
test_data=total_data.iloc[83:127,:] #划分测试集
'''fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(131)
ax1.plot(total_data['Date'],total_data['Close'])
ax2 = fig.add_subplot(132)
ax2.plot(total_data['Date'],total_data['Close_diff1'])
ax3 = fig.add_subplot(133)
ax3.plot(total_data['Date'],total_data['Close_diff2'])
plt.show()'''

#下面是定阶差分，利用ADF检验（单位根检验）数据的平稳性(注意只要看第2个p值即可，如果小于0.05，则说明数据不存在单位根，平稳性检验通过)这个是更加准确判断
diff0_adf=ADF(total_data['Close'])
diff1_adf=ADF(total_data['Close_diff1'])
diff2_adf=ADF(total_data['Close_diff2'])
print('diff0',diff0_adf)
print('diff1',diff1_adf)
print('diff2',diff2_adf)
#由上面的内容可确定差分为2

#平稳性检验通过后进行白噪声检验（只有非白噪声序列才能利用ARIMA模型）(可以看第二个和第4个参数，p值小于0.05则说明其是非白噪声序列，序号表示利用几阶滞后)
res_ljungbox=acorr_ljungbox(total_data['Close_diff2'],lags=24, boxpierce=True, return_df=True) #boxpierce是表示不仅返回QLB统计量检验结果还返回QBP统计量检验结果，return_df表示以dataframe的格式进行返回
print('ljungbox检验',res_ljungbox) #可以看出在1到24阶滞后项均通过了检验，p值小于0.05

#下面进行确定p和q的阶数(注意下面也可以利用原始的所有数据进行操作，并不需要划分训练集和测试集)
#方法1利用acf(自相关系数)和pacf（偏自相关系数)绘制图，利用拖尾和截尾来定阶
plt.figure(figsize=(12,8))
ax1=plt.subplot(211)
sm.graphics.tsa.plot_acf(train_data['Close_diff2'], lags=20,ax=ax1) #注意要利用差分后的数据进行做
ax1.set_xlabel('Order',fontsize=13)
ax1.set_ylabel('Autocorrelation coefficient',fontsize=13)
ax2=plt.subplot(212)
sm.graphics.tsa.plot_pacf(train_data['Close_diff2'], lags=20,ax=ax2)
#自动调节子图之间的间距，防止一些坐标轴标签重合
plt.tight_layout()
ax2.set_xlabel('Order',fontsize=13)
ax2.set_ylabel('Partial autocorrelation coefficient',fontsize=13)
plt.show()

#方法2，利用AIC和BIC准则进行参数估计评价，网格搜索最佳的p和q(最小的BIC则是最佳的)
p_min=0 #AR的最小阶数
p_max=5 #AR的最大阶数
q_min=0 #MA的最小阶数
q_max=5 #MA的最大阶数
d=2 #目前求得的差分阶数
#下面进行构建bic的datarame
bic_dataframe=pd.DataFrame(index=[f'AR{i}' for i in range(p_min,p_max+1)],
                           columns=[f'MA{i}' for i in range(q_min,q_max+1)])
for p,q in itertools.product(range(p_min,p_max+1),range(q_min,q_max+1)): #这个函数的作用就是嵌套for循环
    if p==0 and q==0: #如果AR和MA的阶数都是0的话就没有bic
        bic_dataframe.loc[f'MA{p}',f'MA{q}']=np.nan
        continue
    try:
        model=sm.tsa.ARIMA(train_data['Close_diff2'],order=(p,d,q)) #传入训练数据和参数
        result = model.fit(method='innovations_mle') #训练模型
        bic_dataframe.loc[f'AR{p}',f'MA{q}']=result.aic #计算bic,同理其也可以计算AIC
    except:
        continue
    bic_dataframe=bic_dataframe[bic_dataframe.columns].astype(float) #将数据均变成浮点数
#下面进行绘制bic热点矩阵图
plt.figure(figsize=(8,6))
sns.heatmap(bic_dataframe,
               mask=bic_dataframe.isnull(),
               annot=True,
               fmt='.2f',
               cmap='Spectral') #cmap或者不指定也行
plt.title('BIC',fontsize=15)
plt.show() #从图中我们可以看出最佳的是（p,d,q）为（0，2，3）即MA(3)模型

#下面是更简单的方式得到最佳p和q（利用AIC和BIC）（速度较慢，不推荐）
#train_results = sm.tsa.arma_order_select_ic(train_data['Close_diff2'], ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
#print('AIC', train_results.aic_min_order)
#print('BIC', train_results.bic_min_order)

#下面对模型进行检验（注意对训练集数据进行检验）（残差序列的随机性检验或者也可以检测参数估计的显著性（t检验））残差序列（1阶以后）要全部在2倍标准差以内
model = sm.tsa.ARIMA(train_data['Close_diff2'], order=(0, 2, 3))
results = model.fit()
resid = results.resid #赋值残差序列
plt.figure(figsize=(12,8))
sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
plt.show()

#下面也是对模型进行检验（对残差序列的正态性进行检验）
qqplot(resid, line='q',fit=True)
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.title('QQ-normality test',fontsize=15)
plt.xlabel('Theoretical Quantiles',fontsize=13)
plt.ylabel('Sample Quantiles',fontsize=13)
plt.show()

#首先重新构建全部的数据（将日期作为索引）
new_total_dataframe=pd.DataFrame({'Close_diff2':total_data['Close_diff2'].tolist()},index=total_data['Date'])
#下面进行创建模型（利用差分后的数据）注意是要进行拟合处理后的数据
model=sm.tsa.ARIMA(new_total_dataframe, order=(0, 2, 3)) #设置模型
result=model.fit(method='innovations_mle') #这里是进行模型的拟合过程
#下面是选取一段时间节点进行测试
test_predict=result.predict(start=str('2014-04'),end=str('2014-06-30'),dynamic=False) #04- 06-30
#下面是计算训练集中预测的置信区间（可以进行绘制置信区间图）
resid=result.resid['2014-04':'2014-06-30'] #注意index是带日期的
# 置信系数
conf_level = 0.95
# 残差标准误差
std_error = np.std(resid) #(p+q+d-1)
# 计算置信区间
n = len(new_total_dataframe['Close_diff2'])
df = n - 4 -2 - 0 + 1
t_value = t.ppf(1 - (1 - conf_level) / 2, df)
lower = result.fittedvalues['2014-04':'2014-06-30']  - t_value * std_error #置信区间上界
upper = result.fittedvalues['2014-04':'2014-06-30']  + t_value * std_error #置信区间下界

#将小于0的变为0(如果没有这个需求的话就不用了)
"""new_lower=[]
for i in lower:
    if i>0:
        new_lower.append(i)
    else:
        new_lower.append(0)"""

#下面进行可视化(测试集预测的数据和原始数据进行对比)
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线 
plt.plot(total_data['Date'][62:127],test_predict.tolist(),alpha=0.5,color='#fe0000',label='Predictive value')  #62:127
plt.plot(total_data['Date'][62:127],total_data.iloc[62:127,3].tolist(),alpha=0.5,color='b',label='Original value')
plt.scatter(total_data['Date'][62:127],test_predict.tolist(),alpha=0.5,color='#fe0000',s=10)
plt.scatter(total_data['Date'][62:127],total_data.iloc[62:127,3].tolist(),alpha=0.5,color='b',s=10)
#下面进行绘制95%置信区间
plt.fill_between(total_data['Date'][62:127],lower,upper,alpha=0.3,color='#fe0000',label='95% confidence interval')
plt.legend()
plt.xticks(rotation=30)
plt.xlabel('Date',fontsize=13)
plt.ylabel('Close diff2',fontsize=13)
plt.title('Predicted value vs. original value',fontsize=15)
plt.show()

#下面进行模型的检验（R方等）
#第一种计算R方的方式
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
print('R方',computeCorrelation(test_predict.tolist(),total_data.iloc[62:127,3].tolist())**2) #输出第1种R方指标，注意输入数据一定是可迭代的列表和序列,同时要注意时间要对齐

#下面进行预测后面的值（未知的值）这里是对差分两次的数据进行预测
steps=20 #往后预测的天数
forcast_diff2=result.forecast(steps=steps,alpha=0.05).tolist()#step就是往后预测值的数量，exog可以忽略，置信区间是1-alpha
# 计算置信区间
# 进行 Monte Carlo 模拟，生成未来预测值的分布
num_simulations = 1000  # 假设进行 1000 次 Monte Carlo 模拟
simulations = np.zeros((num_simulations, steps))
for i in range(num_simulations):
    residuals = np.random.choice(result.resid, size=steps)
    simulation = forcast_diff2 + residuals #注意其中的residuals是有正有负的
    simulations[i] = simulation
# 计算置信区间
conf_level = 0.95  # 置信水平
lower_diff2 = np.percentile(simulations, (1 - conf_level) / 2 * 100, axis=0)
upper_diff2 = np.percentile(simulations, (1 + conf_level) / 2 * 100, axis=0)

#注意如果利用了差分还要对最终的预测数据进行还原(这里是对二阶差分进行还原)
forcast_diff1=[] #存储预测的1阶差分数据
forcast_data=[]  #存储预测原始值数据
lower=[]
lower_diff1=[]
upper=[]
upper_diff1=[]

for i in range(len(forcast_diff2)):
    if i == 0:
        #对原始序列的(注意diff1数据的第一个为空值，diff2前两个数据都为空值)
        forcast_diff1.append(total_data['Close_diff1'].tolist()[-1]+forcast_diff2[i])
        forcast_data.append(forcast_diff1[i]+total_data['Close'].tolist()[-1])
        #对下限置信区间
        lower_diff1.append(total_data['Close_diff1'].tolist()[-1] + lower_diff2[i])
        lower.append(lower_diff1[i] + total_data['Close'].tolist()[-1])
        #对上限置信区间
        upper_diff1.append(total_data['Close_diff1'].tolist()[-1] + upper_diff2[i])
        upper.append(upper_diff1[i] + total_data['Close'].tolist()[-1])
    else:
        # 对原始序列的
        forcast_diff1.append(forcast_diff1[i-1]+forcast_diff2[i])
        forcast_data.append(forcast_data[i-1]+forcast_diff1[i])
        # 对下限置信区间
        lower_diff1.append(forcast_diff1[i-1] + lower_diff2[i])
        lower.append(forcast_data[i-1] + lower_diff1[i])
        # 对上限置信区间
        upper_diff1.append(forcast_diff1[i-1]  + upper_diff2[i])
        upper.append(forcast_data[i-1] + upper_diff1[i])

print('forcast_diff2',forcast_diff2) #预测的2阶差分数据
print('forcast_diff1',forcast_diff1) #预测的1阶差分数据
print('forcast_data',forcast_data)   #预测的原始数据

#最后进行绘制预测的图像(需要连接最后一个数据和第一个预测的数据，同时需要再设置一个时间序列跟预测数据相对应的，然后进行绘制即可)
forecast_date=pd.date_range('30/6/2014',periods=steps+1,freq='D') #表示生成2014年从6月30日开始的20天日期数据
forcast_data.insert(0,total_data.iloc[:,1].tolist()[-1]) #在第一个位置插入原始数据的最后一个数，防止绘图有断
forecast_series=pd.Series(forcast_data,index=forecast_date) #将预测数据变成序列
#下面是置信区间的求解

#下面进行绘制图
plt.figure(figsize=(8,6))
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.plot(total_data['Date'],total_data['Close'],alpha=0.5,color='b',label='Original value')
plt.plot(forecast_series,alpha=0.5,color='r',label='Predictive value')
#下面进行绘制置信区间
plt.fill_between(forecast_series.index[1:],lower,upper,alpha=0.3,color='#fe0000',label='95% confidence interval')
#plt.scatter(total_data['Date'],total_data['Close'],alpha=0.5,color='b',s=12)
#plt.scatter(forecast_series,alpha=0.5,color='r',s=12) #目前这个会出现bug
plt.legend()
plt.xticks(rotation=30)
plt.xlabel('Date',fontsize=13)
plt.ylabel('Close',fontsize=13)
plt.title('Forecast Data',fontsize=15)
plt.show()