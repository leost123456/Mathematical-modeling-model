import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize,word_tokenize #分别用于分句和分词
from nltk.corpus import stopwords #用于停用词处理
from gensim.models.ldamodel import LdaModel,CoherenceModel
from gensim import corpora
import pyLDAvis.gensim
from matplotlib import rcParams #导入包

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

#下面进行分词操作，并将所有字母转化成小写字母（结合两步）
def token_data(data): #data为输入的句子列表
    result_list=[] #存储每个句子分词的结果，形式为[[]]
    for sentence in data: #取出每个句子
        sentence=sentence.lower() #将大写字母均转化为小写字母
        result_list.append(word_tokenize(sentence)) #添加分完词的结果
    return result_list

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

#计算困惑度perplexity(困惑度越低越好)
def perplexity(corpus,dictionary,num_topics):
    """
    :param corpus: 语料库（编码后的文本）
    :param dictionary: 构建的词典
    :param num_topics: 主题数量
    :return:
    """
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    return ldamodel.log_perplexity(corpus) #计算困惑都

#计算主题一致性coherence（一致性越高越好）并且一般选用这个作为指标
def coherence(data,corpus,dictionary,num_topics):
    """
    :param data: 经过分词等操作后的文本序列，形式为[[],[]]
    :param corpus: 语料库（编码后的文本）
    :param dictionary: 构建的词典
    :param num_topics: 主题数量
    :return:
    """
    #构建LDA模型
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    #建立CoherenceModel来评估模型的质量和主题之间的连贯性，coherence有定使用哪种一致性得分度量方法，常用的有 'c_v'、'u_mass'、'c_uci'
    ldacm = CoherenceModel(model=ldamodel, texts=data, dictionary=dictionary, coherence='u_mass') #推荐使用u_mass
    return ldacm.get_coherence()

#绘制曲线(用于评估选取主题个数)
def plot_curve(x,y,xlabel,ylabel,title):
    plt.figure(figsize=(8, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    plt.plot(x, y, color='#e74c3c', lw=1.5)
    plt.xlabel(xlabel, fontsize=13, family='Times New Roman')
    plt.ylabel(ylabel, fontsize=13, family='Times New Roman')
    plt.title(title, fontsize=15, family='Times New Roman')
    #plt.savefig('.svg',format='svg',bbox_inches='tight')
    plt.show()

#进行正则表达式，提取出对应的单词和概率
def extra_word_prob(str_data):
    """
    :param str_data: 单个主题的输出分布情况例如‘0.083*"." + 0.042*"ship" + 0.038*"pirates" + 0.036*"," + 0.019*"crew"’
    :return:
    """
    decimal_reg=re.compile(r'(\d*\.\d*)\*') #提取数字部分的正则表达式
    word_reg=re.compile(r'"(\S*)"')
    decimal_ls=[float(decimal) for decimal in decimal_reg.findall(str_data)]
    word_ls=word_reg.findall(str_data)

    return word_ls,decimal_ls #返回词汇序列和其对应的概率序列

#绘制词汇概率分布柱形图
def plot_distribute(word_ls,decimal_ls,color='#e74c3c',title='title'):
    """
    :param word_ls:
    :param decimal_ls:
    :param title: 图标题
    :return:
    """
    index = np.arange(len(word_ls))
    # 进行绘制
    width = 0.25
    plt.figure(figsize=(8, 6))
    plt.tick_params(size=5, labelsize=13)  # 坐标轴
    plt.grid(alpha=0.3)  # 是否加网格线
    # 注意下面可以进行绘制误差线，如果是计算均值那种的簇状柱形图的话(注意类别多的话可以用循环的方式搞)
    plt.bar(index, decimal_ls, width=width,  alpha=0.8,color=color)
    # 下面进行打上标签(也可以用循环的方式进行绘制)(颜色就存储在一个列表中)
    for i, data in enumerate(decimal_ls):
        plt.text(index[i], data + 0.001, round(data, 3), horizontalalignment='center', fontsize=13)
    plt.xlabel('Word', fontsize=13,family='Times New Roman')
    plt.ylabel('Probability', fontsize=13,family='Times New Roman')
    plt.title(title, fontsize=15,family='Times New Roman')
    plt.xticks(index,word_ls,rotation=40)
    #plt.savefig('.svg',format='svg',bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('D:\\2023美赛\\海盗问题\\第二题第三题数据\\EA.csv', encoding='gbk')['Incident details'].to_list()
    #进行分词并去除无用的停用词(输出形式为[[],[]])
    new_data=clean_stopwords(token_data(data))

    # 构建词袋模型(TF表示法,另外的一种方式，利用序号和计数来表示)
    dictionary = corpora.Dictionary(new_data) #构建词典
    corpus = [dictionary.doc2bow(text) for text in new_data] #返回每个text的编码向量

    #利用指标选择合适的主题数（迭代15个主题）(一般用coherence指标)
    num=3
    """perplexity_ls=[perplexity(corpus,dictionary,num_topics=num_topic) for num_topic in range(1,num+1)] #困惑度序列(越小越好)
    coherence_ls = [coherence(new_data,corpus, dictionary, num_topics=num_topic) for num_topic in range(1, num + 1)] #一致度序列（越大越好）

    #绘制迭代曲线（分别两个）
    plot_curve(np.arange(1,num+1),perplexity_ls,'Number of topics','perplexity','Perplexity varies with the number of topics')
    plot_curve(np.arange(1, num + 1), coherence_ls, 'Number of topics', 'Coherence','Coherence varies with the number of topics')
"""
    # 确定主题数量后构建LDA模型
    num_topics = 2  # 设置主题数量
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    #输出每个主题的词分布（每个主题前15个词语）
    topics_words_distribute=lda_model.print_topics(num_words=15)

    #绘制主题-词分布图（以主题1为例）这个是概率的展示类型
    word_ls1,decimal_ls1=extra_word_prob(topics_words_distribute[0][1]) #输出词序列和概率序列
    plot_distribute(word_ls1,decimal_ls1,title='Word probability distribution for topic 1')

    #可视化LDA主题模型结果(保存为html的形式),包括主题间降维后的表示分布，还有各个主体的词分布情况
    result = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(result, r'D:\学习资源\学习资料\算法学习\软件库和基础学习\topic.html')

    #对新文本进行主题的分析（输出新文本的主题分布概率情况）
    new_text=['The pirates took five hostages and fired rockets at the ship']
    new_text=clean_stopwords(token_data(new_text)) #分词和去除停用词
    new_bows = [dictionary.doc2bow(text) for text in new_text] #进行构建词袋编码
    new_topic_distribution = [lda_model.get_document_topics(bow) for bow in new_bows] #输出每个新文本的主题分布
    print(new_topic_distribution)




