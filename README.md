
## Introduction

该仓库记录了我从 TensorFlow 转向 Pytorch 时做的小项目， 主要针对 NLP 初学者，分为三个难度：

- Easy 级别简单实现了一下常见基础模型：逻辑回归， 线性回归， 前馈神经网络， 卷积神经网络，数据集采用 Mnist 数据集。
- Medium 级别针对NLP初学者，采用最简单的文本分类任务， 实现了一些经典模型，如TextCNN， TextRNN, FastText等； 数据集采用sst-2.。
- Hard 级别中实现了一些阅读理解模型，如BiDAF，QANet 等。 这部分难度会很高，阅读理解模型的复杂度应该是NLP任务中最复杂的了，但我人为理解阅读理解领域对NLP的学习是很有帮助的；数据集采用 SQuAD 数据集。

## Requirement

- python 3.6
- numpy
- pytorch = 1.0
- torchvision
- torchtext
- tqdm
- pytorch_pretrained_bert： 可以不安装，直接把github上文件复制过来，这里为了方便，使用pip安装好的来跑

## Easy

```
python RUN_mnist.py --model_name=LR or CNN or FNN
```

| Dataset            | LR   | FNN  | CNN  |
| ------------------ | ---- | ---- | ---- |
| Mnist 手写数字识别 | 92%  | 98%  | 99%  |

## Medium-SST

### SST-2 数据集

数据集源自 [Glue](https://gluebenchmark.com/tasks)， 考虑到 sst-2 并没有给出测试集，因此我将训练集的2000多个样本划分出来形成测试集， 划分代码见：split_train.py 。


### Results

```
python run_SST.py  # 需要在该文件下更改你要运行的模型名字，我将每个模型的参数独立了出来，便于理解调试
```

| model name            | acc    | F1    | loss  |
| --------------------- | ------ | ----- | ----- |
| TextCNN               | 92.53% | 0.925 | 0.195 |
| TextRNN               | 92.13% | 0.924 | 0.207 |
| LSTM_ATT              |   93.07     |  0.930     |  0.285     |

### 关于Bert 

这里参考原始论文仓库中的代码[run_classifier.py](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py)，我对整个代码进行重构，使得其更易于理解，整个结构也更清晰，原论文中的代码要1000+， 而我这里只有300行左右的代码。实现相关的细节如下

- GPU： 1080ti， 11GB， Base 需要 8G 左右显存，Large 需要。 不支持多GPU。
- 关于预训练数据： 如果使用名字如 "bert-base-uncaseed"， 有时会发生加载失败的情况，这是因为如果通过名字搜索，其会先去访问一个网站，而该网站在国内访问会很低下，极有可能会访问失败，从而导致加载失败，因此推荐采用直接指定Bert目录地址的方式。

```
# bert vocabs， 分词的时候会用到
'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",

# pytorch bert 预训练模型参数
'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
```

### Todo

待做： 采用 highway networks 将词向量与char-level向量结合起来