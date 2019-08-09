# NLP_Fundamentals

task - 1

【准备任务】
1.tensorflow安装
推荐Anaconda（针对自己操作系统和位数下载对应版本）；推荐用conda create创建对应的python环境（注：某些python版本可能不支持tensorflow）；通过pip install来安装tensorflow。
参考： 
tensorflow安装教程 
http://www.tensorflownews.com/series/tensorflow-install-tutorial/

2.tensrflow基础
关注图、会话、tensor、变量、feed和fetch；
使用图(graphs)来表示计算任务、在被称之为会话(Session)的上下文(context)中执行图、使用tensor表示数据、通过变量(Variable)维护状态；
使用feed和fetch为任意的操作赋值或者从其中获取数据。
 
参考：
TENSORFLOW从入门到精通之——TENSORFLOW基本操作 http://www.tensorflownews.com/2018/03/28/tensorflow_base/
tensorflow简介 
http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/introduction.html
tensorflow基本使用 http://wiki.jikexueyuan.com/project/tensorflowzh/get_started/basic_usage.html
莫凡tensorflow
 https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
史上最全的Tensorflow学习资源汇总
 https://zhuanlan.zhihu.com/p/35515805
微软周明：自然语言处理的历史与未来
 http://zhigu.news.cn/2017-06/08/c_129628590.htm
【Task1 数据集探索 (2 days)】
1.数据集
数据集：中、英文数据集各一份
中文数据集：THUCNews
THUCNews数据子集：https://pan.baidu.com/s/1hugrfRu 密码：qfud
英文数据集：IMDB数据集 Sentiment Analysis

2.IMDB数据集下载和探索
参考TensorFlow官方教程：
影评文本分类  |  TensorFlow
科赛 - Kesci.com

3.THUCNews数据集下载和探索
参考博客中的数据集部分和预处理部分：
CNN字符级中文文本分类-基于TensorFlow实现 - 一蓑烟雨 - CSDN博客
参考代码：text-classification-cnn-rnn/cnews_loader.py at mas...

4.学习召回率、准确率、ROC曲线、AUC、PR曲线这些基本概念


【打卡链接】
打卡地址：https://shimo.im/sheets/vQHyJW63ydd6HcWW/

【Task2 特征提取 (2 days)】  
1. 基本文本处理技能
1.1 分词的概念（分词的正向最大、逆向最大、双向最大匹配法）；
1.2 词、字符频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）

2. 概念
2.1 语言模型中unigram、bigram、trigram的概念；
2.2 unigram、bigram频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）

3. 文本矩阵化：要求采用词袋模型且是词级别的矩阵化
步骤有：
3.1 分词（可采用结巴分词来进行分词操作，其他库也可以）；
3.2 去停用词；构造词表。
3.3 每篇文档的向量化。

【打卡链接】
打卡地址：
https://shimo.im/sheets/vQHyJW63ydd6HcWW/


