## 简介
我是NLP小白一枚。从1.16开始接触Quora比赛，以下是我对**预训练词向量**的一些归纳~(大部分是照搬kernel的 ...)
- 预训练词向量会带来优势吗？(通过kernel里的例子)
- 使用预训练词向量之前的清洗
- 如何使用预训练词向量

## 1. 预训练词向量真会给你带来额外的优势吗？

![word2vec](https://qph.fs.quoracdn.net/main-qimg-3e812fd164a08f5e4f195000fecf988f)

- 参考的[kernel](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
- 嵌入通常表示基于在文本语料库中一起出现的频率的单词的几何编码。下面描述的word嵌入的各种实现在构造方式上有所不同。

### 1.1 Word2Vec
-  主要思想：在每个单词的**上下文中**训练一个模型，这样类似的单词就会有类似的数字表示。
-  就像一个正常的前馈网络，其中你有一组自变量和一个你试图预测的目标因变量，你首先把你的句子分解成单词(tokenize)，并根据窗口大小创建一些单词对。其中一个组合可以是一对词，如(cat, purr)其中cat是自变量(X) purr是我们要预测的目标因变量(Y)
-  我们将‘cat’通过一个初始化为随机权值的嵌入层送入NN，并将其通过softmax层传递，最终目的是预测‘purr’。优化方法如SGD最小化损失函数“(target word | context words)”，该函数的目的是在给定上下文单词的情况下最小化预测目标单词的损失。如果我们用足够的时间来做这件事，嵌入层中的权值最终将代表单词向量的词汇，也就是单词在这个几何向量空间中的“坐标”。
![Word2Vec示意](https://i.imgur.com/R8VLFs2.png)
- 注：上面的例子采用了skip模型。对于连续单词包(CBOW)，我们基本上是在给定上下文的情况下预测一个单词  

### 1.2 GLOVE

​        GLOVE的工作原理类似于Word2Vec。上面可以看到Word2Vec是一个“预测”模型，它预测给定单词的上下文，GLOVE通过构造一个共现矩阵(words X context)来学习，该矩阵主要计算单词在上下文中出现的频率。因为它是一个巨大的矩阵，我们分解这个矩阵来得到一个低维的表示。有很多细节是相互配合的，但这只是粗略的想法。

### 1.3 FastText
​        FastText与上面的两个嵌入有很大的不同。Word2Vec和GLOVE将每个单词作为最小的训练单元，而FastText使用n-gram字符作为最小的单元。例如，单词vector，“apple”，可以分解为单词vector的不同单位，如“ap”，“app”，“ple”。使用FastText的最大好处是，它可以为罕见的单词，甚至是在训练过程中没有看到的单词生成更好的嵌入，因为n-gram字符向量与其他单词共享。这是Word2Vec和GLOVE无法做到的。

### 1.4 See
实验基于Kaggle的比赛Toxic Comment Classification Challenge，请参见kernel。
我也实验了该kernel，也可以看我的[中文版该kernel](https://github.com/Cooper111/pretrained_embedding_tutorial/blob/master/Toxic%20Comment%20Classification%20Challenge/Do_Pretrained_Embeddings_Give_You_The_Extra_Edge%EF%BC%9F.ipynb)。
注：其中值得一提的技巧
- 该kernel开头给出了另一个keras教程，其中一步，通过查看句子长度图分布来设置maxlen
- 不同的预训练词向量基于不同的语料训练而成，画出训练和验证损失来评估性能
- 使用函数封装制作Embedding_matrix
- 查看预训练词向量包含的喂入文本词汇占总喂入文本词汇的比率或柱状统计

## 2.使用预训练词向量之前的清洗
- 参考的[kernel](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
- 实验基于Kaggle的比赛Quora Insincere Questions Classification，请参见kernel。
我也实验了该kernel，也可以看我的[中文版该kernel](https://github.com/Cooper111/pretrained_embedding_tutorial/blob/master/Quora/how-to-preprocessing-when-using-embeddings.ipynb)。

### 2.1 清洗的两条法则
- 看情况使用stopwords操作，不一定得按照标准的清洗流程
- 使用于训练的语料尽可能接近预训练的语料，就是未登录词尽量少

### 2.2 几个好用的函数
- 计算包含的单词的出现次数，返回w2id的字典
```
函数：
#我将使用下面的函数来跟踪我们的训练词汇，它将遍历我们的所有文本并计算包含的单词的出现次数。
def build_vocab(sentences, verbose = True):
    # 参数  sentences list of list of words，就是二维的
    # 返回值 对应  词和词的次数 的字典
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
使用方法：
#因此，让我们填充词汇表并显示前5个元素及其计数。注意，现在我们可以使用progess_apply查看进度条
sentences = train['question_text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})
```

- 检查词汇表和嵌入之间的交集

```
#接下来，我定义一个函数来检查词汇表和嵌入之间的交集。它将输出一个out of vocabulary (oov)单词列表，我们可以使用它来改进我们的预处理
import operator
def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass
    
    print('Found embeddings for {:.2%} of vocab'.format(len(a) /  len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]#取axis=1维度进行排序，并换为逆序
    
    return sorted_x

使用方法：
oov = check_coverage(vocab, embedding_index)

样例输出：
100%|█████| 253623/253623 [00:00<00:00, 329145.72it/s]
Found embeddings for 57.38% of vocab
Found embeddings for 89.99% of all text

然后可以 oov[:10]这样查看次数靠前的未登录词
```
- 清洗操作

```
train['question_text'] = train['question_text'].progress_apply(lambda x: clean_numbers(x))#progress_apply可以显示进度，tqdm也是
sentences = train['question_text'].progress_apply(lambda x: x.split())#分割英文句子成词汇列表
vocab = build_vocab(sentences)#构建词汇表
```

### 2.3 清洗流程
- ①加载数据，导入预训练词向量
- ②对于Pandas.Series实例应用apply清洗函数
>e.g.   train['question_text'] =  train['question_text'].progress_apply(lambda x: clean_text(x))

- ③构造词汇表
- ④检查词汇表和嵌入之间的交集
- ⑤查看未登录词靠前的几个，构造新的清洗函数
- ②③④⑤循环，例子见2.2清洗操作

### 2.4 清洗思路
- ‘a’,'to'这种停用词在训练GoogleNews预训练词向量训练时被删除了,所以会出现在oov靠前，在最后对用于训练的文本剔除这些词(Kernel里这么做的)
- 特殊符号的去除视情况而定，一些空格替换，一些去除（效果： 未登录词类占总词汇类比76%to43%）
- 数字再处理，预训练词向量里所有大于9的数字都被hashs替换了。即成为# #,123变成# # #或15.80€变成# #,# #€。因此，让我们模拟这个预处理步骤来进一步改进我们的嵌入式覆盖率（效果：未登录词词类占总词汇类 43%to40%）
- 口语用词替换成标准词汇（和思路第一步一起的效果：未登录词量占总词量 11% to 1%）

## 3. 如何使用预训练词向量
- 参考的[kernel](https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings)
- 实验基于Kaggle的比赛Quora Insincere Questions Classification，请参见kernel。
我也实验了该kernel，也可以看我的[中文版该kernel](https://github.com/Cooper111/pretrained_embedding_tutorial/blob/master/Quora/A%20look%20at%20different%20embeddings.!.ipynb)。

### 3.1 使用流程
- 将文本划分训练集和验证集

  ```
  ## split to train and val
  train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)
  ```


- 填充文本缺失值

  ```
  ## fill up the missing values
  train_X = train_df['question_text'].fillna('_na_').values
  val_X = val_df['question_text'].fillna('_na_').values
  test_X = test_df['question_text'].fillna('_na_').values
  ```

  

- 使用Tokenizer讲文本转换为向量序列

  ```
  ## Tokenize the sentences
  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(list(train_X))
  train_X = tokenizer.texts_to_sequences(train_X)
  val_X = tokenizer.texts_to_sequences(val_X)
  test_X = tokenizer.texts_to_sequences(test_X)
  ```

  

- 根据max_len来Pad_sequences,固定句子长度

  ```
  ## Pad the sentences
  train_X = pad_sequences(train_X, maxlen=maxlen)
  val_X = pad_sequences(val_X, maxlen=maxlen)
  test_X = pad_sequences(test_X, maxlen=maxlen)
  ```

  

- 构造词向量嵌入矩阵
  法①

  ```
  EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
  def get_coefs(word, *arr):
      return word, np.asarray(arr, dtype='float32')
  embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8') if len(o)>100 )
  
  all_embs = np.stack(embeddings_index.values())
  emb_mean, emb_std = all_embs.mean(), all_embs.std()
  embed_size = all_embs.shape[1]
  
  word_index = tokenizer.word_index
  nb_words = min(max_features, len(word_index))
  embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
  for word, i in word_index.items():
      if i >= max_features:
          continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
  ```
  法②

  ```
  def loadEmbeddingMatrix(typeToLoad):
      #根据Embedding的不同，从Kaggle加载不同的嵌入文件
      #我们要实验的矩阵
      if(typeToLoad=='glove'):
          EMBEDDING_FILE = '../input/glove.twitter.27B.25d.txt'
          embed_size = 25
      elif(typeToLoad=='word2vec'):
          word2vecDict = word2vec.KeyedVectors.load_word2vec_format('../../Quora/input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
          embed_size = 300
      elif(typeToLoad=='fasttext'):
          EMBEDDING_FILE='../input/wiki.simple.vec'
          embed_size = 300
      
      if(typeToLoad=='glove' or typeToLoad=='fasttext'):
          embeddings_index = dict()
          #通过遍历文件的每一行，将Embedding权重转移到字典中。
          f = open(EMBEDDING_FILE)
          for line in f:
              #split up line into an indexed array
              values = line.split()
              #first index is word
              word = values[0]
              #store the rest of the values in the array as a new array
              coefs = np.asarray(values[1:], dtype='float32')
              embeddings_index[word] = coefs #50 dimensions
          f.close()
          print('Loaded %s word vectors.' % len(embeddings_index))
      else:
          embeddings_index = dict()
          for word in word2vecDict.wv.vocab:
              embeddings_index[word] = word2vecDict.word_vec(word)
          print('Loaded %s word vectors.' % len(embeddings_index))
      
      gc.collect()
      #我们得到嵌入权值的均值和标准差，这样我们就可以保持
      #对于我们自己随机产生的权值的其余部分，同样的统计数据。
      all_embs = np.stack(list(embeddings_index.values()))
      emb_mean, emb_std = all_embs.mean(), all_embs.std()
      
      nb_words = len(tokenizer.word_index)
      #我们讲设置Embedding的尺寸为我们复制的预训练的维度
      #Embedding矩阵大小将为 词汇中词数 X Embedding Size
      embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
      gc.collect()
      
      #使用新创建的嵌入矩阵，我们将用两个矩阵中的单词填充它
      #我们自己的字典和加载预训练嵌入。
      embeddedCount = 0
      for word, i in tokenizer.word_index.items():
          i -= 1
          #然后我们看看这个词是否在GLove的字典里，如果是，得到相应的权重
          embedding_vector = embeddings_index.get(word)
          #并存储在Embedding矩阵中，我们稍后将用其进行训练。
          if embedding_vector is not None:
              embedding_matrix[i] = embedding_vector
              embeddedCount += 1
      print('total embedded:', embeddedCount,'common words')
      
      del(embeddings_index)
      gc.collect()
      
      #最后，返回Embedding矩阵
      return embedding_matrix
      
  ```

  

- 训练模型

  ```
  model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
  ```

  

- 得到验证样本预测和F1得分的最佳阈值,得到测试集预测

  ```
  pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)
  for thresh in np.arange(0.1, 0.501, 0.01):
      thresh = np.round(thresh, 2)
      print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
   #得到测试集预测
   pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
  ```

  

- 清理一些内存

  ```
  del model, inp, x
  import gc
  gc.collect()
  time.sleep(10)
  ```



## 3.2 词向量优化方法
- 通过统一模型对不同预训练词向量和无预训练词向量，绘制损失曲线来选取
- 尝试给定各个结果以权重进行ensemble


## TODO
- 学习大佬的总结~
- 复现大佬的代码~
- 吸取建议，然后补全词向量优化方法~
- 祝各位比赛取的好成绩，我先去旅游到28号再继续···


<hr>
学号： 071-沈凯文