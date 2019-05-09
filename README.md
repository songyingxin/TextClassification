
## Introduction

该仓库记录了我从 TensorFlow 转向 Pytorch 时做的小项目， 主要针对 NLP 初学者，分为三个难度：

- Easy 级别简单实现了一下常见基础模型：逻辑回归， 线性回归， 前馈神经网络， 卷积神经网络。
- Medium 级别针对NLP初学者，采用最简单的文本分类任务， 实现了一些经典模型，如TextCNN， TextRNN, LSTM+Attentioon, RCNN等。
- Hard 级别中实现了一些阅读理解模型。 这部分难度会很高，阅读理解模型的复杂度应该是NLP任务中最复杂的了，但我认为理解阅读理解领域对NLP的学习是很有帮助的。

本仓库主要实现Bert之前相关的模型，如果想看看 Bert 之后的相关实现，可以看看我其他仓库。

## Models

**对于模型，我大多都在[我的博客](https://www.zhihu.com/people/songyingxin/posts)内做了详细的介绍， 简单模型一笔略过， 复杂模型往往都单独独立一篇文章**

## Requirement

- python 3.6
- numpy
- pytorch = 1.0
- torchvision
- torchtext
- tqdm

## 数据集

本仓库分别采用三个数据集： Mnist 手写数字识别， SST-2情感分类，RACE阅读理解数据集。
其中， SST-2数据集来自[Glue](https://gluebenchmark.com/tasks)， 考虑到 SST-2 并没有给出测试集，因此我将训练集的2000多个样本划分出来形成测试集。
RACE 数据集可以从 [RACE](http://www.qizhexie.com//data/RACE_leaderboard)下载。之所以不采用 squad 是后来考虑到 squad 太过繁琐，其实不易于理解，使用。

不过，推荐从我的百度云下载，后续我会给出连接。


## Easy

```
python RUN_mnist.py --model_name=LR or CNN or FNN
```

| Dataset            | LR   | FNN  | CNN  |
| ------------------ | ---- | ---- | ---- |
| Mnist 手写数字识别 | 92%  | 98%  | 99%  |

## Medium-SST

**注意：** 由于这是实验性质的数据，因此没有死扣参数细节，主要专注于模型的实现，且考虑到词向量的方式可能将被淘汰，因此许多小的Trick 就没有实现，如 Highway Networks 等。

```
python run_SST.py  # 需要在该文件下更改你要运行的模型名字，我将每个模型的参数独立了出来，便于理解调试
```

| model name            | acc    | F1    | loss  |
| --------------------- | ------ | ----- | ----- |
| TextCNN               | 92.53% | 0.925 | 0.195 |
| TextRNN               | 92.13% | 0.924 | 0.207 |
| LSTM_ATT              |   93.07     |  0.930     |  0.285     |
| TextRCNN  | 94.06 | 0.940 | 0.165


## Hard-RC

- 一个小tirck： 复杂模型常采用 char-level + word-level 来获得词的最终embedding， 考虑到 Bert 终结了此道， 因此，感觉无甚必要去实现这些细节，因此，此时只使用了 word-level 来作为模型的 Embedding 层。



## Reference Papers

[1] TextCNN： Convolutional Neural Networks for Sentence Classification

[3] A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

[4] Recurrent Convolutional Neural Network for Text Classification

[5] Hierarchical Attention Networks for Document Classification

[n] Large Scale Multi-label Text Classification With Deep Learning