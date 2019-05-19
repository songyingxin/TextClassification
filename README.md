
## Introduction

该仓库记录了我从 TensorFlow 转向 Pytorch 时做的小项目， 主要针对 NLP 初学者，分为三个难度：

- Easy 级别简单实现了一下常见基础模型：逻辑回归， 线性回归， 前馈神经网络， 卷积神经网络。
- Medium 级别针对NLP初学者，采用文本分类任务， 实现了一些经典模型，如TextCNN， TextRNN, LSTM+Attentioon, RCNN， Transformer 等。
- Hard 级别中实现了一些阅读理解模型。 这部分难度会很高，阅读理解模型的复杂度应该是NLP任务中最复杂的了，但我认为理解阅读理解领域对NLP的学习是很有帮助的。

本仓库主要实现Bert之前相关的模型，如果想看看 Bert 之后的相关实现，可以看看我其他仓库。

## Models

**对于模型，我大多都在[我的博客](https://www.zhihu.com/people/songyingxin/posts)内做了详细的介绍， 简单模型一笔略过， 复杂模型往往都单独独立一篇文章。**

最近，为了测试 `Highway Networks` 在连接词向量上的表现， 添加了通过 `Highway Networks` 融合 `char-level` 向量和 `word-level` 向量， 主要与之前的模型进行对比。 

关于 RACE 数据集的模型， 后续会逐渐完善，主要是模型较多， 且现在侧重点放在了 Bert 之上， 更新可能会慢一些。

## Requirement

- python 3.6
- numpy
- pytorch = 1.0
- torchvision
- torchtext
- tqdm
- tensorboardx

## 数据集

本仓库分别采用三个数据集： Mnist 手写数字识别， SST-2情感分类，RACE阅读理解数据集。

- SST-2数据集来自[Glue](https://gluebenchmark.com/tasks)， 考虑到 SST-2 并没有给出测试集，因此我将训练集的2000多个样本划分出来形成测试集。
- RACE 数据集可以从 [RACE](http://www.qizhexie.com//data/RACE_leaderboard)下载。之所以不采用 squad 是后来考虑到 squad 太过繁琐，其实不易于理解，使用。

不过，推荐从我的百度云下载，百度云连接为： 

```
sst-2: 链接：https://pan.baidu.com/s/1ax9uCjdpOHDxhUhpdB0d_g     提取码：rxbi 
RACE: 待更新
```


## Easy

```
python RUN_mnist.py --model_name=LR or CNN or FNN
```

| Dataset            | LR   | FNN  | CNN  |
| ------------------ | ---- | ---- | ---- |
| Mnist 手写数字识别 | 92%  | 98%  | 99%  |

## Medium-SST

**注意：** 由于这是实验性质的数据，因此没有死扣参数细节，主要专注于模型的实现。

### Train and Test

```
# word-level
python run_SST.py --do_train --epoch_num=10   # train and test
python run_SST.py   # test

# char-level + word-level
python run_Highway_SST.py --do_train --epoch_num=10   # train and test
python run_Highway_SST.py  # test

tensorboard --logdir=.log   # 可视化分析
```

### Results

| model name            | acc    | F1    | loss  |
| --------------------- | ------ | ----- | ----- |
| TextCNN               | 92.53 | 92.5 | 0.195 |
| TextRNN               | 92.13 | 92.4 | 0.207 |
| LSTM_ATT              |   93.07     |  93.0    |  0.285     |
| TextRCNN | 94.06 | 94.0 | 0.165 |
| TextCNNHighway | 93.21 | 93.21 | 0.185 |
| TextRNNHighway | 92.03 | 92.03 | 0.199 |
| LSTMATTHighway | 93.02 | 93.01 | 0.278 |
| TextRCNNHighway | 93.266 | 93.37 | 0.167 |
### Analysis

由上表可以看出， 复杂模型要比简单模型表现好， 从 loss 曲线上看， 加上 `Highway Networks` 效果普遍会好，但并没有使得模型表现突飞猛进， 可能是由于SST数据集所用词汇相对简单的原因，OV问题并不严重 。


## Hard-RC



## Reference Papers

[1] TextCNN： Convolutional Neural Networks for Sentence Classification

[2] A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

[3] Recurrent Convolutional Neural Network for Text Classification

[4] Hierarchical Attention Networks for Document Classification

[5] Large Scale Multi-label Text Classification With Deep Learning