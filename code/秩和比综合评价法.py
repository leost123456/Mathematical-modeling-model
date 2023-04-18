import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams #导入包
from scipy.stats import levene #T检验和方差齐性检验
from statsmodels.formula.api import ols #单因素方差分析
from statsmodels.stats.anova import anova_lm

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#下面进行构建秩和比综合评价的函数，其中data是dataframe的格式，weight是传入的权重列表（可以用其他赋权算法搞，否则就是等权）,
# threshold是最后进行分级的Probits范围，列表形式例如[2, 4, 6, 8][(2, 4] < (4, 6] < (6, 8]]，默认分成3级，full_rank表示是否是整秩算秩秩，否则用非整秩的算法，一般是非整秩
def rsr(data, weight=None, threshold=None, full_rank=False):
    Result = pd.DataFrame() #构建一个dataframe数据结构用于存储原始数据和对应的秩、RSR、Probit、RSR Regression和level
    n, m = data.shape #得到数据的行列数

    # 对原始数据编秩
    if full_rank: #选择整秩算法
        for i, X in enumerate(data.columns): #对每列原始数据
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
            Result[f'R{str(i + 1)}：{X}'] = data.iloc[:, i].rank(method="average") #注意原值越大，排名的数值越大
    else:#采用非整秩算法
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
            Result[f'R{str(i + 1)}：{X}'] = 1 + (n - 1) * (data.iloc[:, i]-data.iloc[:, i].min()) / (
                        data.iloc[:, i].max() - data.iloc[:, i].min())

    # 计算秩和比
    weight = 1 / m if weight is None else np.array(weight) / sum(weight) #加入权重
    Result['RSR'] = (Result.iloc[:, 1::2] * weight).sum(axis=1) / n #计算RSR值
    Result['RSR_Rank'] = Result['RSR'].rank(ascending=False) #计算RSR_rank的值,这里是RSR越大排名的数值越小（1）这个数据意义不大

    # 绘制 RSR 分布表
    RSR = Result['RSR']
    RSR_RANK_DICT = dict(zip(RSR.values, RSR.rank().values))#存储RSR和其排名的一个字典元组序列，注意RSR越大，其排名的数值越大
    print(RSR_RANK_DICT)
    #下面进行构建分布的表格
    Distribution = pd.DataFrame(index=sorted(RSR.unique()))#先按照RSR进行升序排列，index
    Distribution['f'] = RSR.value_counts().sort_index() #进行计算RSR的频数
    Distribution['Σ f'] = Distribution['f'].cumsum() #计算累计频数
    Distribution['R-'] = [RSR_RANK_DICT[i] for i in Distribution.index] #得到平均秩次(也就是其原始的一个秩次)
    Distribution['R-/n*100%'] = Distribution['R-'] / n  # 根据平均秩次计算累计频率
    Distribution.iat[-1, -1] = 1 - 1 / (4 * n)  # 修正最后一项累计频率
    Distribution['Probit'] = 5 - stats.norm.isf(Distribution.iloc[:, -1])  # inverse survival function 将累计频率换算为概率单位

    # 计算回归方差并进行回归分析（下面是进行一元线性回归）RSR=a*Probit+b(r0为一个元组，第一个值为斜率，第二个为截距)
    r0 = np.polyfit(Distribution['Probit'], Distribution.index, deg=1)  # x,y

    #下面的这个也是进行一元线性回归结果和上面的相同(最小二乘法得到)，主要为了得到resid进行后续的残差正态检验，检测回归模型的有效性
    model = sm.OLS(Distribution.index, sm.add_constant(Distribution['Probit']))
    result = model.fit()
    print(result.summary()) #各种资料打印，还有各种评价模型的指标展示，R方、AIC，BIC等

    # 残差检验，检验残差是否符合正态分布，从而证明模型的有效性，z为统计量，p为p值，p小于0.05则认为拒绝原假设，认为不符合正态分布
    z_error, p_error = stats.normaltest(
        result.resid.values)  # tests the null hypothesis that a sample comes from a normal distribution
    print(f"残差分析：统计量为{z_error} p值为{p_error}")

    #输出回归方程
    if r0[1] > 0:
        print(f"\n回归直线方程为：y = {r0[0]} Probit + {r0[1]}")
    else:
        print(f"\n回归直线方程为：y = {r0[0]} Probit - {abs(r0[1])}")

    #下面进行绘制拟合曲线
    x=Distribution['Probit']
    y1=Distribution.index #原始的RSR值
    y2=np.polyval(r0,Distribution['Probit'])#修正后的RSR值（线性曲线）
    plt.figure(figsize=(8,6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线·
    plt.plot(x,y1,label='Original value',marker='o',color='#bf0000')
    plt.plot(x, y2, label='Corrected value', marker='*', color='#F21855')
    #打上标签
    # 下面进行打上标签
    for i, data in enumerate(y1):
        plt.text(x.tolist()[i], data + 0.002, round(data, 2), horizontalalignment='center', fontsize=13)
    for i, data in enumerate(y2):
        plt.text(x.tolist()[i], data + 0.002, round(data, 2), horizontalalignment='center', fontsize=13)
    plt.legend()
    plt.xlabel('Probit',fontsize=13)
    plt.ylabel('RSR', fontsize=13)
    plt.title('Original value and corrected RSR', fontsize=15)
    plt.show()

    # 代入回归方程并分档排序
    Result['Probit'] = Result['RSR'].apply(lambda item: Distribution.at[item, 'Probit']) #找到Distribution中RSR(index)对应的Probit
    ##下面得到修正后的RSR（将Probit带入回归方程得到）
    Result['Regression'] = np.polyval(r0, Result['Probit'])
    #下面进行分级,先得到修正后RSR范围
    threshold = np.polyval(r0, [2, 4, 6, 8]) if threshold is None else np.polyval(r0, threshold)
    #根据范围将修正后的RSR进行分级（注意这里是修正后的RSR越大，则等级的数值越大
    Result['Level'] = pd.cut(Result['Regression'], threshold,
                             labels=range(1,len(threshold)))  # Probit分组[(2, 4] < (4, 6] < (6, 8]]

    #下面利用单因素方差分析来检验分档结果的有效性
    # 首先进行方差齐性检验
    if len(threshold)-1==3: #分成3级的情况
        test1 = levene(Result[Result['Level']==1]['Regression'],Result[Result['Level']==2]['Regression'],
                       Result[Result['Level']==3]['Regression'])
    elif len(threshold)-1==4: #分成4级的情况
        test1 = levene(Result[Result['Level']==1]['Regression'],Result[Result['Level']==2]['Regression'],
                       Result[Result['Level']==3]['Regression'],Result[Result['Level']==4]['Regression'])
    elif len(threshold) - 1 == 5:  # 分成5级的情况
        test1 = levene(Result[Result['Level'] == 1]['Regression'], Result[Result['Level'] == 2]['Regression'],
                       Result[Result['Level'] == 3]['Regression'], Result[Result['Level'] == 4]['Regression'],
                       Result[Result['Level'] == 3]['Regression'])
    #下面输出结果
    print(f'方差齐性检验的统计量为{test1[0]},p值为{test1[1]}')  # 第一个是统计量，第二个是p值，p值大于0.05说明方差具有齐性

    #方差齐性检验如果通过可以进行单因素方差分析
    model = ols('Regression ~ C(Level)',Result).fit()  # 前面的第一个是数值型变量，第二个是分类型变量，要加C，其表示Catagory
    #F值越大，p值越小，说明两者之间差异越显著
    anovaResults = anova_lm(model)  # 看输出中分类变量的p值和F值即可
    print('单因素方差分析结果为：',anovaResults)

    return Result,Distribution #最终输出原来的结果表，和分布表

# 读取数据
data1 = pd.DataFrame({'产前检查率': [99.54, 96.52, 99.36, 92.83, 91.71, 95.35, 96.09, 99.27, 94.76, 84.80],
                     '孕妇死亡率': [60.27, 59.67, 43.91, 58.99, 35.40, 44.71, 49.81, 31.69, 22.91, 81.49],
                     '围产儿死亡率': [16.15, 20.10, 15.60, 17.04, 15.01, 13.93, 17.43, 13.89, 19.87, 23.63]},
                    index=list('ABCDEFGHIJ'), columns=['产前检查率', '孕妇死亡率', '围产儿死亡率'])

#因为下面两个是成本型指标，需要反向排序(进行正向化操作，注意对结果不影响，因为这个算法主要看排名)
data1["孕妇死亡率"] = 1 / data1["孕妇死亡率"]
data1["围产儿死亡率"] = 1 / data1["围产儿死亡率"]
# 下面是调用子函数进行秩和比几个步骤的计算
a,b=rsr(data1,weight=[0.33,0.33,0.33],threshold=[2,4,6,8],full_rank=False) #注意其中的threshold是设定的Probit范围进行分级的
#print(a['Regression']) #输出修正后的RSR(也就是综合的评分)
#print(a['Level']) #输出分级结果