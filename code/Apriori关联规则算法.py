import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 准备数据(可以用于01数据或者分类数据（将其转化为01多个01变量）)，注意可以用定义字符串的形式表现
dataset = [['A', 'B', 'C'],
           ['A', 'B', 'D', 'E'],
           ['B', 'E', 'F']]

# 转换数据格式
te = TransactionEncoder() #创建转换数据格式的对象
te_ary = te.fit(dataset).transform(dataset) #转换
df = pd.DataFrame(te_ary, columns=te.columns_) #其实就是每个01变量用True或者false的数据格式（可以自己构建）
print(df)

# 使用Apriori算法找到频繁项集（出现频率高的，先利用各个项集出现的支持度进行筛选）返回的为dataframe格式
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# 生成关联规则，可以利用置信度或者提升度进行筛选出关联规则（返回的为dataframe格式）
# 注意其中的metrix包含的指标有："support"（支持度）"confidence"（置信度）"lift"（提升度）"leverage"（杠杆）"conviction"（确信度）具体公式科参考文档
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)

#进行保存关联规则数据
rules.to_csv('',index=None)

# 输出频繁项集和关联规则
print("频繁项集：")
print(frequent_itemsets)
print("\n关联规则：")
print(rules)  #第一个参数表示前项，第二个参数表示后项（对于计算支持度、提升度来说没有先后区分关系），输出筛选后项集的各种指标包括支持度、置信度、提升度等等
