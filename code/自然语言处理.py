import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from nltk.tokenize import sent_tokenize,word_tokenize #分别用于分句和分词
from nltk.corpus import stopwords #用于停用词处理
import string
import nltk
from nltk.stem import WordNetLemmatizer #用于词性还原
from nltk.stem import PorterStemmer #用于词干提取
from nltk.stem.lancaster import LancasterStemmer #用于词干提取
from nltk.stem.snowball import SnowballStemmer #用于词干提取（推荐）
import operator
from matplotlib import rcParams #导入包
from imageio import imread
from wordcloud import WordCloud
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer #词袋模型运用
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF模型算法运用
import gensim
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer #情感分析器
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#下面先进行导入数据
data1=pd.read_csv('D:\\2023美赛\\海盗问题\\第二题第三题数据\\EA.csv',encoding='gbk')

#下面进行分词操作，并将所有字母转化成小写字母（结合两步）
def token_data(data): #data为输入的句子列表
    result_list=[] #存储每个句子分词的结果，形式为[[]]
    for sentence in data: #取出每个句子
        sentence=sentence.lower() #将大写字母均转化为小写字母
        result_list.append(word_tokenize(sentence)) #添加分完词的结果
    return result_list
token_data1=token_data(data1['Incident details']) #最终的形式为[[]]

#下面是进行去除无用的停用词
def clean_stopwords(data): #data表示分完词后的列表形式为[[]]
    result_list=[] #存储最后的结果，注意最后的形式为[[]]
    sr=stopwords.words('english') #存储所有停用词的列表
    for inner_data in data: #取出大列表中包含的小列表（分完词的列表）
        mid_list=[] #存储一个句子中去除所有停用词的词
        for token in inner_data: #取出单个词
            if token not in sr:
                mid_list.append(token)
        result_list.append(mid_list)
    return result_list
clean_data1=clean_stopwords(token_data1) #最终的形式为[[]]

#下面进行去除所有的标点符号
def clean_punctuation(data):#data是分完词的列表，形式为[[]]
    punctuation_string=["'",'!','"','#','$','%','&','\\','(',')','*','+',',','-','.','/',':',';','<',
                        '=','>','?','@','[',']','^','_','`','{','|','}','~',"'s",'’','0'] #存储所有的英文标点的列表
    result_list=[] #存储最终的结果
    for inner_data in data: #取出一个句子的所有词
        mid_list=[]
        for token in inner_data: #取出词
            if token not in punctuation_string:
                mid_list.append(token)
        result_list.append(mid_list)
    return result_list
clean_data1=clean_punctuation(clean_data1) #最终的形式为[[]]

#下面进行词干提取
def stemming_data(data): #data表示分完词的列表,形式为[[]]
    result_list=[] #存储最终的结果，形式为[[]]
    stemmer1 = PorterStemmer() #波特词干提取器  (偏宽松) 推荐
    stemmer2 = LancasterStemmer  # 朗卡斯特词干提取器   (偏严格)
    stemmer3 = SnowballStemmer # 思诺博词干提取器   (偏中庸) 推荐
    for inner_data in data:
        mid_list=[]
        for token in inner_data:
            mid_list.append(stemmer1.stem(token))
        result_list.append(mid_list)
    return result_list
stemming_data1=stemming_data(clean_data1)

#下面进行词性标注并进行词形还原（注意和词干提取两个只要用一种就行了，一般用词形还原）
def lemmatization_data(data): #data表示分完词的列表,形式为[[]]
    result_list=[] #存储最终的结果，形式为[[]]
    lemmatizer = WordNetLemmatizer() #创建词形还原的对象
    for inner_data in data: #取出每一条记录的所有分词列表
        pos_tagged = nltk.pos_tag(inner_data) #进行词性标注,注意是列表的形式，类型为[(),()],[('word', 'NN'), ('better', 'RBR'), ('had', 'VBD')]
        mid_list=[]
        for token,tag in pos_tagged: #取出一个个词和其标注的词性
            if tag.startswith('NN'): #如果是名词的话
                mid_list.append(lemmatizer.lemmatize(token,'n'))
            elif tag.startswith('JJ'): #如果是形容词的话
                mid_list.append(lemmatizer.lemmatize(token,'a'))
            elif tag.startswith('VB'): #如果是名词的话
                mid_list.append(lemmatizer.lemmatize(token,'v'))
            elif tag.startswith('RB'): #如果是副词的话
                mid_list.append(lemmatizer.lemmatize(token,'r'))
            else: #如果没有词性的话九用原来的词
                mid_list.append(token)
        result_list.append(mid_list)
    return result_list
lemmatization_data1=lemmatization_data(clean_data1)

#下面的是词频统计（注意一定要用分词后的数据，经过以上处理后的数据也行）
def word_count(data,n): #其中data是输入数据，形式为[[]]，n为取前n个高词频的词数据进行绘制图像
    freq_dis={} #创建储存结果的字典
    for inner_data in data:
        for token in inner_data:
            if token in freq_dis: #如果已经有了
                freq_dis[token]+=1
            else: #如果没有
                freq_dis[token]=1
    #下面进行排序的操作(降序，并选取前n个),输出为[(),()]的形式，第一个数为词，第二个为出现的次数
    sort_data=sorted(freq_dis.items(),key=operator.itemgetter(1),reverse=True)[:n] #注意字典用items()后返回一个发生器，要调用后面的函数才能进行排序，降序
    #下面进行分别获取词名称序列、词频统计序列和累加序列
    word_list=[x[0] for x in sort_data]  #词序列
    count_list=[x[1] for x in sort_data] #词频统计序列
    cumcum_list=[] #累加序列
    k=0
    for i,count in enumerate(count_list):
        if i==0:
            k=count
        else:
            k+=count
        cumcum_list.append(k)
    #下面进行绘制柱形图和折线图（双坐标轴）
    index = np.arange(n)
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(8,6)) #设置第一个图
    ax1.tick_params(size=5, labelsize=13)  # 坐标轴
    ax1.grid(alpha=0.3)  # 是否加网格线
    #下面进行绘制柱形图
    bar1=ax1.bar(index,count_list,width=width,color='b',alpha=0.4,label ='Word frequency')
    ax1.set_xlabel('word', fontsize=13)  # 设置x轴参数
    ax1.set_ylabel('Counts', fontsize=13)  # 设置y轴的名称
    ax1.set_title('Word frequency statistics', fontsize=15)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=12)  # 设置第一个类型数据y轴的颜色参数，同时设置y轴的字体大小
    plt.legend(loc='upper left')
    plt.xticks(index, word_list, rotation=30) #注意这里要进行设置x轴标签，后面设置会有问题
    #下面进行绘制累计的折线图
    ax2 = ax1.twinx()  # 和第一类数据共享x轴（新设置图）
    plot1 = ax2.plot(index,cumcum_list, color='r', alpha=0.5, marker='o', lw=2, label='Add up')
    ax2.set_ylabel('Cumulative count', fontsize=13)  # 设置ax2的y轴参数
    ax2.tick_params(axis='y', labelcolor='r', labelsize=12)  # 设置ax2的y轴颜色,同时设置y轴的字体大小
    plt.legend(loc='upper right')
    # 下面进行打上标签
    for i, data in enumerate(count_list):
        ax1.text(index[i], data + 2, round(data, 3), horizontalalignment='center', fontsize=10)
    for i, data in enumerate(cumcum_list):
        ax2.text(index[i], data + 5, round(data, 0), horizontalalignment='center', fontsize=10)
    plt.show()
#下面是输出结果
#word_count(lemmatization_data1,15)

#下面进行绘制词云图
#下面的是可以将已经文本预处理后的数据进行合并，搞成一个个句子在一个列表中
def trans_string1(data): #data是输入数据形式为[[]]
    result_list=[]
    for inner_data in data:
        result_list.append(' '.join(inner_data))
    return result_list #输出形式为[]
trans_data1=trans_string1(lemmatization_data1)

#先将所有数据合并到一个字符串(构建函数)这个是全部搞成一个字符串
def trans_string2(data): #data是输入数据形式为[[]]
    result = ''
    for inner_data in data: #最外层的，包含了四个海域
            result = result +' '+ ' '.join(inner_data) #注意要加空格，以空格拆分
    return result
#下面进行输出结果，已经是一个字符串的形式
trans_data=trans_string2(lemmatization_data1)

#下面进行绘制
"""mask=Image.open('D:\\2023美赛\\海盗问题\\海盗船图片模板1.png') #读取蒙版图片（注意一般要读取白底、分辨率高的图片）
mask_array=np.array(mask) #要将图片转换成数组的形式
#下面进行创建词云图对象
wc=WordCloud(#背景颜色
             background_color='white',
             #设置背景宽
             width=1000,
             #设置背景高
             height=800,
             #最大字体
             max_font_size=1000,
             #最小字体
             #min_font_size=10,
             #显示的最大单词数量
             max_words=200,
             #mode='RGBA',
             #使用的颜色图
             #colormap='Greens', 
             #设置背景蒙版
             mask=mask_array)
#下面进行输入数据，注意输入的是一条字符串
wc.generate(trans_data)
plt.figure('pac_negtive') #绘图名称
#以图片形式显示词云
plt.imshow(wc)
#关闭图像坐标系
plt.axis('off')
plt.savefig('D:\\2023美赛\\海盗问题\\图\\detail词云图.svg',format='svg',bbox_inches='tight')
plt.show()
"""

#下面进行文本文本向量化操作
#1.one-hot编码方式
#实验的语料
corpus = ['My dog has a flea problems.',
          'Maybe it is stupid to take him to a dog park.',
          'Try to prevent my dog from eating my steak.']

#构建数据中所有单词的索引
def one_hot(data): #data是输入的数据，形式为[[]]，也就是已经过了文本预处理后的数据
    # 构建数据中所有单词的索引
    token_index = {}
    for inner_data in data:
        for word in inner_data:
            if (word not in token_index) and (len(token_index)==0):
                token_index[word] = 0
            elif (word not in token_index) and (len(token_index)!=0):
                token_index[word] = len(token_index)
    #设置每个句子取的最大单词数量
    max_length=len(token_index)
    #生成矩阵(三维矩阵)
    matrix = np.zeros((len(data), max_length, max(token_index.values())+1))
    for i,inner_data in enumerate(data): #每个句子
        for j,word in enumerate(inner_data): #每个词
            index=token_index[word] #取出词的序号
            matrix[i,j,index]=1
    return token_index,matrix #返回词的序列号字典和文本向量矩阵（三维数据，第一维为各个句子，第二维为每个词，第三维数据为各个词的0-1编号）
#输出数据
token_index,token_matrix=one_hot(lemmatization_data1)

#2.词袋模型（统计各个词在句子中出现的次数，二维数据输出）
def bag_words(data): #data是输入的数据，形式为[]，其中为每个句子，注意这个句子也是可以先进行预处理后拼接而成的
    # 先对输入的数据进行整合成一个个字符串,输出形式为[]
    new_data = trans_string1(data)
    # 实例化
    vectorizer = CountVectorizer()
    # 生成词汇表
    term_frequencies = vectorizer.fit_transform(new_data)
    vocab = vectorizer.get_feature_names() #这个是词汇的名称列表
    #下面进行输出词频矩阵
    term_frequencies_matrix = term_frequencies.toarray()
    #注意下面还可以尝试用词袋模型对新的语句进行向量化
    #vectorizer.transform(['My dog likes steak.']).toarray()
    return vocab,term_frequencies_matrix #输出词汇表和文本向量（二维形式，第一维表示各个句子，第二维表示各个句子中的词频向量）
#输出最终的结果
vocab1,term_frequencies=bag_words(lemmatization_data1)

#3.TF-IDF模型
def TF_IDF(data):#其中输入数据data的形式为[[]]
    #先对输入的数据进行整合成一个个字符串,输出形式为[]
    new_data = trans_string1(data)
    # 实例化
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(new_data)
    # 得到词汇表
    vocab = tfidf_vectorizer.get_feature_names()
    # 得到TF-IDF权重矩阵
    tfidf_matrix = tfidf.toarray()
    return vocab,tfidf_matrix #返回词汇表和权重矩阵（二维形式，第一维表示各个句子，第二维表示各个句子中的词权重）
#输出数据
vocab2,tfidf_matrix=TF_IDF(lemmatization_data1)

#4.N-gram（注意其结合了TF-IDF的一个思想）
def N_gram(data,n):#其中输入数据data的形式为[[]],n表示滑动窗口的步长，一般选择2或者3
    # 先对输入的数据进行整合成一个个字符串,输出形式为[]
    new_data = trans_string1(data)
    # 实例化 (ngram_range=(1,1) 表示 unigram, ngram_range=(2,2) 表示 bigram, ngram_range=(3,3) 表示 thirgram)
    ngram_vectorizer = TfidfVectorizer(ngram_range=(n, n))
    ngram = ngram_vectorizer.fit_transform(new_data)
    # 得到词汇表
    vocab = ngram_vectorizer.get_feature_names()
    # 计算TF-IDF权重矩阵
    ngram_matrix = ngram.toarray()
    return vocab,ngram_matrix #输词组和权重矩阵（二维形式，第一维为各个句子，第二维为各个句子中各词组的权重矩阵）
#输出数据
vocab3,ngram_matrix=N_gram(lemmatization_data1,2)

#词向量编码（对词进行编码，利用神经网络模型可以得到，可以用于词的分类、相似度比较、寻找相似词）
#首先进行创建模型，注意要用分完词后的数据，形式为[[]],模型主要需要调节的参数就是sg,window,min_count和workers
model=gensim.models.Word2Vec(lemmatization_data1,sg=1,vector_size=100,window=5,min_count=1,negative=3,sample=0.001,hs=1,workers=4,epochs=5)
#sg=1是skip—gram算法，对低频词敏感，默认sg=0为CBOW算法
#size是隐含层神经元个数，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。注意有多少个神经元就输出多少维的向量
#window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）
#min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
#negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3,
#negative: 如果>0,则会采用negativesamping，用于设置多少个noise words
#hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
#workers是线程数，此参数只有在安装了Cpython后才有效，否则只能使用单核
#epochs表示训练的轮数

#下面是保存训练后的模型
model.save('word2vec') #保存在文件中
#下面是加载模型
model1=Word2Vec.load('word2vec')
#下面是进行追加训练
#model1.train(more_sentence)

#下面是输出部分
#1.计算一个词最近似的n个词
similar_word=model.wv.most_similar('pirate',topn=10) #前10个
#2.计算两个词的相似度（余弦）
similarity=model.wv.similarity('pirate','guard')
#获取词向量(100维，根据隐含层神经元的数量)
word_vec=model.wv['pirate']

#注意我们可以将一个句子中所有的词向量进行取平均得到句子的向量（这个向量就可以作为后面的机器学习或者神经网络的输入部分实现分类）
def sentance_vec(data): #data是输入的数据，形式为[[]]
    result_list=[]
    for inner_data in data:
        mid_vec=0
        for word in inner_data:
            mid_vec+=model.wv[word]
        result_list.append(mid_vec/len(inner_data))
    return np.array(result_list) #输出所有的句子向量,是narray的形式，shape是二维的（a,b）第一维就是各个句子，第二维就是各个句子的向量
#下面输出最终结果
sentance_vec1=sentance_vec(lemmatization_data1)

#下面利用nltk自带的情感分析器进行情感分析
sid=SentimentIntensityAnalyzer() #创建情感分析器对象
pos_list=[]#存储正向得分
neg_list=[]#存储负向得分
neu_list=[]#存储中立得分
compound_list=[]#存储复杂度
for sen in trans_data1: #注意输入数据的形式为[]，其中为每个句子（经过预处理然后合并过的，不过用原始语句也行）
    ss=sid.polarity_scores(sen) #得到情感得分,其中包括负向得分、中立得分、正向得分，三者得分和为1，还有一个复杂度，是字典的一个形式{'neg': 0.233, 'neu': 0.767, 'pos': 0.0, 'compound': -0.7184}
    pos_list.append(ss['pos'])
    neg_list.append(ss['neg'])
    neu_list.append(ss['neu'])
    compound_list.append(ss['compound'])
#下面可以绘制四个的一个分布图（用散点图或者分类堆积直方图均可）,还可以给出一些语句的情感分析的例子搞个表格，左边为句子，右边为情感分析的结果
#首先处理数据，整理成一个dataframe
#将所有名称和数据分别整理到一个序列中
total_name=['pos' for x in range(len(pos_list))]+['neg' for x in range(len(pos_list))]+['neu' for x in range(len(pos_list))]+['compound' for x in range(len(pos_list))]
total_list=pos_list+neg_list+neu_list+compound_list
#下面将其搞进一个dataframe中
test_dataframe=pd.DataFrame()
test_dataframe['name']=total_name
test_dataframe['score']=total_list
#下面进行绘制分类堆积直方图
sns.set_theme(style="ticks") #设置一种绘制的主题风格
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)
sns.histplot(
    test_dataframe,
    x="score", hue="name", #注意hue是分类的标签，里面可以是分类的标签（字符和数值型均可）,同时数据标签的前后关系是按照读取数据的先后关系的
    multiple="stack",
    palette="light:m_r",
    edgecolor=".3",
    linewidth=.5,
    log_scale=False,
)
plt.tick_params(size=5,labelsize = 13)  #坐标轴
plt.grid(alpha=0.3)                     #是否加网格线
plt.ylabel('Counts',fontsize=13,family='Times New Roman')
plt.xlabel('Score',fontsize=13,family='Times New Roman')
plt.title('Distribution of Sentiment Analysis Results',fontsize=15,family='Times New Roman')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#ax.set_xticks(np.linspace(0.1,5.1,5)) #设置x轴的标签
plt.show()

#下面进行文本分类操作（利用机器学习算法（朴素贝叶斯、svm、集成模型）或者神经网络进行分类操作）
#先进行伪造标签数据
label=np.random.randint(0,5,size=sentance_vec1.shape[0]) #生成与句子数量相等的标签0~4
#生成二值化的标签数据
new_label=label_binarize(label,classes=[0,1,2,3])
#进行划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(sentance_vec1,new_label, test_size=0.2,random_state=42)
y_trans_test=np.argmax(y_test,axis=1) #将one-hot类型的测试集标签转化一下，后面便于绘制

#下面进行建立硬投票模型
RF=RandomForestClassifier(n_estimators=100,random_state=20) #随机森林模型
knn=KNeighborsClassifier(3) #knn
svc=SVC(C=0.6,break_ties=False,cache_size=200,gamma=2000,probability=True)#svc
gassuan=GaussianNB() #高斯贝叶斯
decision_tree=DecisionTreeClassifier() #决策树模型

#建立硬投票模型
weights=np.array([0.25,0.25,0.25,0.25]) #由于权重都是相等的因此可以认为是硬投票模型
vote1=VotingClassifier(estimators=[('Randomforest',RF),('KNN',knn),
                                   ('SVC',svc),('gassuan',gassuan),
                                   ('decision_tree',decision_tree)],voting='soft') #voting=soft or hard
# Learn to predict each class against the other(其是一种多类分类的策略们可以为每一种类被配备一个分类器，01)
classifier = OneVsRestClassifier(vote1,n_jobs=-1) #建立多类分类器模型用，用硬投票模型
#进行训练模型(目前也可运用投票模型（软投票)
classifier.fit(x_train,y_train)
y_score=classifier.predict(x_test)  #预测各个类别[0 0 0 1]
y_predict=np.argmax(y_score,axis=1) #将one-hot的编码格式转化成分类标签，作为真正的预测分类
y_proba=classifier.predict_proba(x_test) #计算每个类别概率（注意此时相加总和不一定是1）

#下面开始计算每一类的ROC（测试集的）
n_classes=new_label.shape[1] #类别数
fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(n_classes): #遍历类别
    fpr[i],tpr[i], _ = metrics.roc_curve(y_test[:, i], y_proba[:, i]) #计算x轴和y轴的值
    roc_auc[i] = metrics.auc(fpr[i], tpr[i]) #计算auc面积值
# Compute micro-average ROC curve and ROC area（方法二:将每个类别原始值和预测值都进行展平再进行计算ROC）
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_proba.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

#绘制ROC曲线
plt.figure(figsize=(8,6))
#绘制平均的ROC曲线
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue','g'] #颜色序列有几类用几种
for i ,color in enumerate(colors):
    plt.plot(fpr[i],tpr[i],color=color,lw=2,label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i])) #绘制曲线同时打上AUC面积标签
plt.plot([0, 1], [0, 1], 'k--', lw=2)  #绘制对角线
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC-Kurve',fontsize=15)
plt.legend(loc="lower right")
plt.show()

#下面进行绘制混淆矩阵
confusion_matrix=metrics.confusion_matrix(y_trans_test,y_predict)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
plt.xlabel('prediction category',fontsize=13) #横轴是预测类别
plt.ylabel('real category',fontsize=13)       #数轴是真实类别
plt.title('confusion matrix',fontsize=15)
plt.show()

#下面进行计算训练集交叉验证的平均精度（处理方式就是利用重新划分数据集，用原始作为标签）
x_train, x_test, y_train, y_test = train_test_split(sentance_vec1,label, test_size=0.2,random_state=42)
vote1.fit(x_train,y_train)
#下面是得到交叉验证的综合准确率
score=cross_val_score(vote1,x_train,y_train)
print('交叉验证综合准确率:',score.mean())
#下面是测试集的准确率
print('硬投票测试集精度',metrics.accuracy_score(y_test,y_predict)) #计算测试集的准确率(软投票)

