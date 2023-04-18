import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams
from statsmodels.tsa.stattools import adfuller as ADF

#先设置绘图的全局字体类型
config = {"font.family":'Times New Roman'}  # 设置画图字体类型
rcParams.update(config)    #进行更新配置

#下面进行导入数据集
data=pd.read_csv('D:\\2023美赛\\海盗问题\\deal_data.csv',encoding='gbk')

#时间转换操作
data['Date']=pd.to_datetime(data['Date'])

#时间筛选
new_data=data[(data['Date']>='2008-5') & (data['Date']<='2020-5')] #可以加上多重筛选利用&和|符号进行

#数据重采样操作（包括升采样和降采样）原本1天的变为统计3天的就是降采样，反之为升采样
#下面是统计各日、周、月、年海盗事件发生次数的函数（注意输入的只需要时间序列，当然如果想将次数换成其余变量只需要改变'num'的值 即可，）
def count_data(data):#其中data表示为一个时间的序列，注意也可以为一个dataframe，但是需要改变下函数，注意如果需要用其余的统计的话就将sum()改成其余函数即可
    date_data = pd.DataFrame(data)  # 单独的时间数据
    date_data['num'] = 1  # 新设置一列，用于后续统计次数
    date_data = date_data.set_index(date_data['Date'])  # 注意要将时间设置为索引才可以进行后续的重采样
    date_D=date_data.resample('D').sum() #统计每天发生的海盗事件次数
    date_W=date_data.resample('W').sum() #统计每周发生的海盗事件次数
    date_M=date_data.resample('M').sum() #统计每月发生的海盗事件次数
    date_Y=date_data.resample('Y').sum() #统计每年发生的海盗事件次数
    return date_D,date_W,date_M,date_Y
data_D,data_W,data_M,data_Y=count_data(data['Date']) #输出

#插值方式（填补空值的方式）
data_D1=data_D.ffill(1) #取前面的一个数的值
data_D2=data_D.bfill(1) #取后面一个数的值
data_D3=data_D.interpolate('linear') #利用线性拟合填充
data_D4=data_D.interpolate('quadratic') #二次插值
data_D5=data_D.interpolate('cubic') #三次插值

#滑动窗口进行统计指标（一般搭配绘制图片）
new_data_D1=data_D1.rolling(window=10).mean() #还可以使用sum()、std()等等

#时间数据的平稳性检测（单位根检验）(注意只要看第2个p值即可，如果小于0.05，则说明数据不存在单位根，平稳性检验通过)这个是更加准确判断
adf=ADF(data_D['num'])

#对数变换(注意原始数值不能出现小于等于0的数)
data_log=np.log(data_Y['num'])

#差分（1阶和2阶，一般就是这两种）剔除周期性影响和使时间序列平稳最常用的操作
diff_1=data_Y.diff(1)
diff_2=data_Y.diff(2)

#平滑法（包括移动平滑法和各种指数平滑法）
size=20
#1移动平均法
rol_mean = data_M.rolling(window=size).mean()
#2加权移动平均(对时间距离目前越近的原始数据赋予越高的权重，注意权重的总和为1（加权）,再进行移动平均)注意可以调整参数halflife、com、span、alpha来调节权重
rol_weighted_mean=data_M.ewm(halflife=size,min_periods=0,adjust=True,ignore_na=False).mean()
#3指数平滑法(注意只要调整adjust为False即可，调整权重可以用三个参数为：halflife、span、com和alpha具体公式见文档)
exponential_weight_mean=data_M.ewm(span=2,adjust=False,ignore_na=False).mean()

#时间序列的季节性分解
#其中输入数据注意index是时间格式(直接输入应该也没什么大问题)，period是设置的周期（用于消除季节性因素影响）model的选择有加法和乘法两种类型{“additive”, “multiplicative”}，注意选择
result=seasonal_decompose(data_M['num'],period=12,model='additive') #注意选择周期后会出现一些空值，可以用前面的方法进行填补空值
trend=result.trend #趋势因子
seasonal=result.seasonal #季节性因子
resid=result.resid #残差因子
#下面进行绘图
fig,ax=plt.subplots(3,1,figsize=(8,6)) #设置画布，前面两个参数表示行列
#trend
ax[0].plot(trend.index,trend,label='Trend',color='#F21855',lw=1.5,alpha=1)
#下面进行添加阴影
ax[0].fill_between(trend.index,0,trend,alpha=0.3,color='#F21855')
#下面添加箭头表示趋势
ax[0].annotate('', xy=(trend.index[-15], 25), xytext=(trend.index[-40],30),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2,alpha=0.6))
ax[0].set_ylabel('Trend',fontsize=13)
ax[0].set_title('Time series decomposition results',fontsize=15)
ax[0].legend()
ax[0].tick_params(size=5,labelsize = 13) #坐标轴
ax[0].grid(alpha=0.3)                    #是否加网格线
#seasonal
ax[1].plot(trend.index,seasonal,label='Seasonality',marker='o',color='#52D896',markersize=2,lw=1.5,alpha=1)
ax[1].set_ylabel('Seasonality',fontsize=13)
ax[1].legend()
ax[1].tick_params(size=5,labelsize = 13) #坐标轴
ax[1].grid(alpha=0.3)                    #是否加网格线
#resid
ax[2].scatter(trend.index,resid,label='Residuals',marker='o',color='#FFDD24',s=15,alpha=1)
#在y=0处添加一条辅助线
plt.axhline(y=0, color='#bf0000',lw=2,linestyle='dashed')
ax[2].set_xlabel('Date',fontsize=13)
ax[2].set_ylabel('Residuals',fontsize=13)
ax[2].legend()
ax[2].tick_params(size=5,labelsize = 13) #坐标轴
ax[2].grid(alpha=0.3)                    #是否加网格线
plt.tight_layout() #自动调整子图间距
plt.show()

#后面进行提取傅里叶系数（进行傅里叶变换）