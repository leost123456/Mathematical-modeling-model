#非线性目标函数的一般解法（利用scipy进行求解）
'''result1=''
for i in range((len(t_list)-1)//2):
    # 计算最终的预测影长与真实影长之间的差值的绝对值
    a1 = f'np.arcsin(np.sin({Declination_1} * np.pi / 180) * np.sin(x[1] * np.pi / 180) + np.cos({Declination_1} * np.pi / 180) * np.cos(x[1] * np.pi / 180) * np.cos(15 * ({t_list[i + 10]} + (x[0] - 300) / 15) * np.pi / 180))' # 注意这里取的是后10天，这样可以增大偏差
    # 也是计算太阳高度角（i*2）
    a2 = f'np.arcsin(np.sin({Declination_1} * np.pi / 180) * np.sin(x[1] * np.pi / 180) + np.cos({Declination_1} * np.pi / 180) * np.cos(x[1] * np.pi / 180) * np.cos(15 * ({t_list[i]} + (x[0] - 300) / 15) * np.pi / 180))'
    # 计算方位角（i*2+1）
    o1 = f'np.arccos((np.sin({Declination_1} * np.pi / 180) * np.cos(x[1] * np.pi / 180) - (np.cos({Declination_1} * np.pi / 180) * np.sin(x[1] * np.pi / 180) * np.cos(15 * ({t_list[i + 10]} + (x[0] - 300) / 15) * np.pi / 180))) / np.cos(a1))'
    # 计算方位角（i*2）
    o2 = f'np.arccos((np.sin({Declination_1} * np.pi / 180) * np.cos(x[1] * np.pi / 180) - (np.cos({Declination_1} * np.pi / 180) * np.sin(x[1] * np.pi / 180) * np.cos(15 * ({t_list[i]} + (x[0] - 300) / 15) * np.pi / 180))) / np.cos(a2))'

    l = f'np.abs(np.tan({a2})/np.tan({a1})-{l_list6[i+10]}/{l_list6[i]})'
    if i!=(len(t_list)-1)//2-1:
        result1+=l+'+'
    else:
        result1+=l
#定义最终的目标函数
def fun():
    v=lambda x:eval(result1)
    return v

#创建限制函数(注意这里的限制条件不足，还需要多设置几个)
def con():
    cons=[]
    for t0 in t_list: #注意当type为ineq的时候，默认后面的约束时<=0，如果是等式就将type变为eq。
        cons.append({'type': 'ineq', 'fun': lambda x: eval(f'x[2]/np.tan(np.arcsin(np.cos(({t0}+(x[0]-300)/15) * np.pi / 180) * np.cos({Declination_1} * np.pi / 180) * np.cos(x[1] * np.pi / 180) + np.sin({Declination_1} * np.pi / 180) * np.sin(x[1] * np.pi / 180)))')})
        cons=tuple(cons)
        return cons
cons=con() #得到约束条件

#定义初始预测值(经度、纬度、杆高)进行循环遍历取最小目标值，得到最终的经度和纬度（要通过不断的设置初始值来最终得到全局最优解而不是局部最优）
#x0就是初始值
bounds=((0,None),(0,None),(0,None)) #求解参数的范围
new_x_list=[]
result_list=[]
for i in range(180):
    for j in range(90):
        x0=np.asarray((i,j))
        res = minimize(fun(), x0, method='SLSQP', bounds=bounds,constraints=cons) #bounds就是需要规划求解的参数的范围
        result_list.append(res.fun) #目标函数值
        new_x_list.append(res.x) #求解的参数结果
print(new_x_list[np.argmin(result_list)])'''

#用其他解法来求解非线性规划问题（其准确度有待探究）
import cvxpy as cp #只适合解决参数比较少，比较好手输入的式子，不适合约束条件负责或者目标函数很长的式子，不能用eval()，或者直接手输也行
#同时其适合解决0-1规划的问题
def target_func4(x,length,index_list,risk_list): #length是长度这里是99，index_list是评级列表，返回的是字符串类型的函数
    result=0 #注意后面是求最小的，因此加个负号求最大
    a=0
    for i in range(length):
        if index_list[i]==3:#评级为A时
            a=(1-risk_list[i])*x[i*2]*x[i*2+1]*(1-(-5095*x[i*2]**4+2574*x[i*2]**3-519.3*x[i*2]**2+52.64*x[i*2]-1.41)) #注意这里还需修改多项式回归函数
        elif index_list[i]==2:#评级为B时
            a = (1-risk_list[i])*x[i * 2]*x[i * 2 + 1]*(1-(-6769*x[i*2]**4+3378*x[i*2]**3-649.2*x[i*2]**2+60.68*x[i*2]-1.589))
        elif index_list[i]==1: #评级为C时
            a = (1-risk_list[i])*x[i * 2]*x[i * 2 + 1]*(1-(-2202*x[i*2]**4+1340*x[i*2]**3-320.1*x[i*2]**2+38.5*x[i*2]-1.098))
        if i != length - 1:
            result += a
        else:
            result+=a
    result=-(result)
    return result

def money_count1(x,length):
    total_money=0
    for i in range(length):
        if i!= length-1:
            total_money+=x[i*2+1]
        else:
            total_money+=x[i*2+1]
    return total_money

length_y=15
n=15
x=cp.Variable(n) #设置n个变量
#创建目标函数
target_y=target_func4(x,)
obj=cp.Minimize(target_y)
#创建约束条件
total_money_y=money_count1(x,length_y)
cons=[-total_money_y+9900>=0,total_money_y-990>=0]
prob=cp.Problem(obj,cons)
#进行求解
ans=prob.solve(solver='CPLEX') #注意CPLEX是专门进行求解非线性问题的
print(ans)
print(x.value)

#遍历的方式求解目标规划问题（其实也就是找到目标函数最小的参数）
#定义已知或者固定变量
'''data_1=pd.read_csv('D:\\2022国赛数模练习题\\2022宁波大学数学建模暑期训练第三轮训练题B\\附件1.csv',encoding='gbk')
#计算影子长度并构建影子长度序列
data_1['影子长度']=(data_1['x坐标(米)'].apply(lambda x: x**2)+data_1['y坐标(米)'].apply(lambda x: x**2))**0.5
l_list6=[x for x in data_1['影子长度']]
x_list=[x for x in data_1['x坐标(米)']] #x坐标参数
y_list=[x for x in data_1['y坐标(米)']] #y坐标参数
N=108 #第108天
t_list=[14.7+i*0.05 for i in range(21)] #时间序列,转化成小时来表示
#计算赤纬角
Declination_1=23.45*np.sin(2*np.pi*(284+N)/365) #跟日期有关

total_error_list=[] #误差序列
latitude_list=[] #纬度序列
longitude_list=[] #经度序列
for i in range(0,180): #经度
    for j in range(0,90): #纬度
        error=0 #天数累积的误差
       # for w in range(1,101):
           # new_w=w*0.01
          #  print(new_w)
        for step in range((len(t_list)-1)//2): #这个是时间的步长序列
            # 计算太阳高度角（弧度制）i*2+1
            a1 = np.arcsin(np.sin(Declination_1* np.pi / 180)*np.sin(j* np.pi / 180)+np.cos(Declination_1* np.pi / 180)*np.cos(j* np.pi / 180)*np.cos(15*(t_list[step+10]+(i-300)/15)*np.pi/180)) #注意这里取的是后10天，这样可以增大偏差
            # 也是计算太阳高度角（i*2）
            a2 = np.arcsin(np.sin(Declination_1* np.pi / 180)*np.sin(j* np.pi / 180)+np.cos(Declination_1* np.pi / 180)*np.cos(j* np.pi / 180)*np.cos(15*(t_list[step]+(i-300)/15)*np.pi/180))
            # 计算方位角（i*2+1）
            o1 = np.arccos((np.sin(Declination_1* np.pi / 180)*np.cos(j* np.pi / 180)-(np.cos(Declination_1* np.pi / 180)*np.sin(j* np.pi / 180)*np.cos(15*(t_list[step+10]+(i-300)/15)*np.pi/180)))/np.cos(a1))
            # 计算方位角（i*2）
            o2 = np.arccos((np.sin(Declination_1* np.pi / 180)*np.cos(j* np.pi / 180)-(np.cos(Declination_1* np.pi / 180)*np.sin(j* np.pi / 180)*np.cos(15*(t_list[step]+(i-300)/15)*np.pi/180)))/np.cos(a2))
            #计算偏差影长误差
            error += np.abs(np.tan(a2)/np.tan(a1)-l_list6[step+10]/l_list6[step])
            #计算方位角偏差注意这里还可以取权重来减少误差
            #error+=np.abs(np.abs(o1-o2)-np.abs(np.arccos((x_list[step+10]*x_list[step]+y_list[step+10]*y_list[step])/(l_list6[step+10]*l_list6[step]))))
        total_error_list.append(error)

print(np.argmin(total_error_list))
print(min(total_error_list))
print(f'经度为{np.argmin(total_error_list)//90}')
print(f'纬度为{np.argmin(total_error_list)%90}')
'''

#线性规划就是可以用pulp
"""for i in range(24):
    prob1=pulp.LpProblem('problem2',sense=pulp.LpMinimize)
    #进行批量定义决策变量
    var1=[pulp.LpVariable(f'x{x}',0,round(6000*avg_rate_23[x][i],0),pulp.LpInteger) for x in range(23)]
    #添加目标函数
    prob1+=pulp.lpDot(np.array(total_parameter[i*12:(i+1)*12]),var1)
    #添加约束条件
    prob1+=(pulp.lpDot(np.array(original_product23),var1)>=40000)
    prob1.solve() #求解最优解
    #输出
    #print('Status:',pulp.LpStatus[prob1.status]) #输出求解状态
    for i in prob1.variables():
        result_list1.append(i.varValue)
        print(i.name, '=', i.varValue)  # 输出每个变量的最优解
    print('F(x)', '=', pulp.value(prob1.objective))  # 输出最优解的目标函数值

    result_func1.append(pulp.value(prob1.objective))"""
