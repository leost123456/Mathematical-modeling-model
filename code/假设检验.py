import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind,levene,ttest_rel #T检验和方差齐性检验
from scipy import stats
import statsmodels.stats.weightstats as sw
from  scipy.stats import chi2_contingency,chisquare # 卡方检验
from scipy.stats import fisher_exact #fisher精确检验
from statsmodels.formula.api import ols #单因素方差分析
from statsmodels.stats.anova import anova_lm #anova_lm用于一个或多个因素的方差分析,analysis of variance_linear models
import numpy as np
from statsmodels.graphics.api import qqplot
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data_hpj1=pd.DataFrame(pd.read_excel('D:\\2022国赛数模练习题\\1组红葡萄.xlsx'))
data_hpj2=pd.DataFrame(pd.read_excel('D:\\2022国赛数模练习题\\2组红葡萄.xlsx'))
data_hpj1.drop('组别',axis=1,inplace=True)
data_hpj2.drop('组别',axis=1,inplace=True)

#第一种正态性检验方式：进行KS正态性检验(注意原假设是符合正态分布)
hpj1_columns=data_hpj1.columns
hpj2_columns=data_hpj2.columns
statistic1=[]
p_value1=[]
judge1=[]
for name in hpj1_columns: #选择那一组
    u=data_hpj1[name].mean()#均值
    std=data_hpj1[name].std() #方差
    a=stats.kstest(data_hpj1[name],'norm',(u,std))#主要的正态分布检验代码，p值大于0.05说明其符合正态分布
    statistic1.append(a[0])
    p_value1.append(a[1])
    if a[1] > 0.05 :
        judge1.append('yes')
    elif  a[1] <= 0.05 :
        judge1.append('no')
data1=pd.DataFrame({'statistic':statistic1,'p_value1':p_value1,'yes or no':judge1})
#data1.to_csv('D:\\2022国赛数模练习题\\红葡萄1组正态分布检验.csv',index=None)
#第二种正态性检验方式：绘制QQ图
qqplot(data_hpj1[hpj1_columns[0]], line='q', fit=True)
plt.tick_params(size=5,labelsize = 13) #坐标轴
plt.grid(alpha=0.3)                    #是否加网格线
plt.title('QQ-normality test',fontsize=15)
plt.xlabel('Theoretical Quantiles',fontsize=13)
plt.ylabel('Sample Quantiles',fontsize=13)
plt.show()

#下面是进行方差齐性检验和T检验
statistic2=[]
statistic3=[]
p_value2=[]
p_value3=[]
judge2=[]
judge3=[]
#一定要进行方差齐性检验，因为方差齐不齐性会对t检验的自由度计算产生影响
for name in hpj1_columns:
    result=levene(data_hpj1[name],data_hpj2[name]) #主要的F检验算法代码
    if  result[1]>0.05:#判断是否齐性(方差齐次性检验)，注意零假设是方差相等，备选假设是方差不相等
        judge2.append('yes')
        t_result=ttest_ind(data_hpj1[name],data_hpj2[name]) #主要的t检验算法代码
        statistic3.append(t_result[0])
        p_value3.append(t_result[1])
        #判断独立样本是否显著（注意零假设是相等，备选假设是不相等）
        if t_result[1]<=0.05:
            judge3.append('yes')
        else:
            judge3.append('no')
    # 进行独立样本t检验如果方差不齐性还要注意
    else:
        judge2.append('no')
        t_result1=ttest_ind(data_hpj1[name],data_hpj2[name],equal_var=False)
        statistic3.append(t_result1[0])
        p_value3.append(t_result1[1])
        if t_result1[1]<=0.05: #注意这里需要去除程序才能正常运行
            judge3.append('yes')
        else:
            judge3.append('no')
    statistic2.append(result[0])
    p_value2.append(result[1])

#配对样本T检验
x = [20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21.0, 21.2]
y = [17.7, 20.3, 20.0, 18.8, 19.0, 20.1, 20.0, 19.1]
# 配对样本t检验
print('配对样本',ttest_rel(x, y))

"""data2=pd.DataFrame({'statistic':statistic2,'p_value1':p_value2,'yes or no':judge2})
data2.to_csv('D:\\2022国赛数模练习题\\两组红葡萄方差齐性检验.csv',index=None)
data3=pd.DataFrame({'statistic':statistic3,'p_value3':p_value3,'yes or no':judge3})
data3.to_csv('D:\\2022国赛数模练习题\\红葡萄独立样本t检验结果.csv',index=None)
"""

#Z检验（U检验）(用于总体方差已知，零假设是与均值相等，备选假设是与均值不相等)
arr=[23,36,42,34,39,34,35,42,53,28,49,39,46,45,39,38,
     45,27,43,54,36,34,48,36,47,44,48,45,44,33,24,40,50,32,39,31]
#看数据的均值是否与39相等(双侧)，如果要知道该样本的均值是否大于39，则可以设置alternative="smaller"，零假设就是小于等于39
#输出有两个参数，第一个是统计量Z的值，第二个参数是p值
print(sw.ztest(arr, value=39))

#卡方检验独立性检验（零假设是两者独立，不相关，备选假设是两者不独立，存在相关性）用于两组分类变量
#首先读取数据
data=pd.read_csv('D:\\Desktop\\2022数模国赛\\支撑材料\\data\\处理后的附件1.csv',encoding='gbk')
#下面进行生成列链表（交叉表）n大于等于40，且所有频数E大于等于5
cross_data1=pd.crosstab(data['纹饰'],data['表面风化']) #表面风化为行，纹饰作为列，这里表示风化是否与纹饰有关系
kf = chi2_contingency(cross_data1,False) #卡方检验主体代码部分,False表示不用yates校正卡方
print('chisq-statistic=%.4f, p-value=%.4f, df=%i \n expected_frep: \n%s'%kf) #第一个参数是统计量卡方的值，第二个是p值，第三个是自由度，最后一个是理论期望值

#当样本量n大于等于40，但是频数E小于等于5大于等于1的量超过25%时利用yates校正卡方
kf1 = chi2_contingency(cross_data1,True) #卡方检验主体代码部分
print('yates修正卡方：chisq-statistic=%.4f, p-value=%.4f, df=%i \n expected_frep: \n%s'%kf) #第一个参数是统计量卡方的值，第二个是p值，第三个是自由度，最后一个是理论期望值

#当样本量n小于40，且存在频数小于1则用fisher精确检验（fisher卡方）(注意目前python只能实现2*2的，要实现2*c可以利用spsspro)
c_static,p_value=fisher_exact(np.array([[1,4],[5,9]]),alternative='greater')
print(f'fisher精确检验：chisq-statistic={c_static},p_value={p_value}',) #第一个参数是统计量卡方的值，第二个是p值，第三个是自由度，最后一个是理论期望值

#卡方拟合优度检验（用于一组分类变量）原假设是没有显著差异，备选假设是有显著差异
test_data=np.array([29,31])
c_statistic,p=chisquare(test_data)
print(f'卡方拟合优度检验 ：chisq-statistic={c_statistic},p_value={p},') #其中chisq-statistic是统计量，p_value是p值

#方差分析（用于分类变量和数值型变量，不符合正态分布或者方差略有不齐对结果的影响不大）
#首先先读取数据
data=pd.read_csv('D:\\Desktop\\2022数模国赛\\支撑材料\\data\\预测后的数据.csv',encoding='gbk')
data_K=data[data['类型']=='高钾']
data_B=data[data['类型']=='铅钡']
#首先进行正态性检验
def normality_test(data): #data是一个列表数据
    mean=np.mean(data)
    std=np.std(data)
    result=stats.kstest(data,'norm',(mean,std)) #主要的正态分布检验代码，p值大于0.05说明其符合正态分布
    return result[0],result[1] #第一个是统计量，第二个是p值
print(normality_test(data_K['氧化铅(PbO)']))
print(normality_test(data_B['氧化铅(PbO)']))
#下面进行方差齐性检验（可以将所有类别的数据依次进行输入，比如在后面还可以再加data_B['氧化铜(CuO)]'）
result=levene(data_K['氧化铅(PbO)'],data_B['氧化铅(PbO)'])
print('方差齐性检验',result[0],result[1]) #第一个是统计量，第二个是p值，p值大于0.05说明方差具有齐性
#下面进行单因素方差分析（第一种）F值越大，p值越小，说明两者之间差异越显著
F_statistic, p = stats.f_oneway(data_K['氧化铅(PbO)'],data_B['氧化铅(PbO)'])

#第二种方式（注意如果是多因素方差分析，则在后面继续添加即可，例如'height ~ C(water) + C(sun) + C(water):C(sun)'，其中最后一个C代表的是交互作用）
data.rename(columns={'类型':'type','氧化铅(PbO)':'PbO','颜色':'color'},inplace=True) #注意第二种方式输入要是英文的
model = ols('PbO ~ C(type)', data).fit() #前面的第一个是数值型变量，第二个是分类型变量，要加C，其表示Catagory
anovaResults = anova_lm(model) #看输出中分类变量的p值和F值即可
print(anovaResults)

#如果方差分析显著下面就进行多重比较，明确是哪两个因素之间有显著差异(用tukey方法)
print(pairwise_tukeyhsd(data['PbO'],data['type'])) #如果reject为True则说明拒绝原假设，认为两个类别之间有显著差异

#秩和检验（用于数据分布未知，各种参数也未知的情况,当p值小于0.05时，说明两者具有显著差异，总体分布不同）
result1=stats.ranksums(data_B['氧化铅(PbO)'],data_K['氧化铅(PbO)']) #用于两类
result2=stats.kruskal(data_B['氧化铅(PbO)'],data_K['氧化铅(PbO)']) #用于两类及以上
print(result1,result2)
