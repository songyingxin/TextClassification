
## Introduction

该仓库记录了我从 TensorFlow 转向 Pytorch 时做的小项目， 主要针对 NLP 初学者，分为三个难度：

- Easy 级别简单实现了一下常见基础模型：逻辑回归， 线性回归， 前馈神经网络， 卷积神经网络。
- Medium 级别针对NLP初学者，采用最简单的文本分类任务， 实现了一些经典模型，如TextCNN， TextRNN, LSTM+Attentioon等。
- Hard 级别中实现了一些阅读理解模型，如BiDAF，QANet 等。 这部分难度会很高，阅读理解模型的复杂度应该是NLP任务中最复杂的了，但我人为理解阅读理解领域对NLP的学习是很有帮助的。

本仓库主要实现Bert之前相关的模型，如果想看看 Bert 之后的相关实现，可以看看我其他仓库。

## Requirement

- python 3.6
- numpy
- pytorch = 1.0
- torchvision
- torchtext
- tqdm

## 数据集

本仓库分别采用三个数据集： Mnist 手写数字识别， SST-2情感分类，SQuAD阅读理解数据集。
其中， SST-2数据集来自[Glue](https://gluebenchmark.com/tasks)， 考虑到 SST-2 并没有给出测试集，因此我将训练集的2000多个样本划分出来形成测试集。
SQuAD 数据集可以从 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)下载。

不过，推荐从我的百度云下载，后续我会给出连接。


## Easy

```
python RUN_mnist.py --model_name=LR or CNN or FNN
```

| Dataset            | LR   | FNN  | CNN  |
| ------------------ | ---- | ---- | ---- |
| Mnist 手写数字识别 | 92%  | 98%  | 99%  |

## Medium-SST

```
python run_SST.py  # 需要在该文件下更改你要运行的模型名字，我将每个模型的参数独立了出来，便于理解调试
```

| model name            | acc    | F1    | loss  |
| --------------------- | ------ | ----- | ----- |
| TextCNN               | 92.53% | 0.925 | 0.195 |
| TextRNN               | 92.13% | 0.924 | 0.207 |
| LSTM_ATT              |   93.07     |  0.930     |  0.285     |

## Hard-RC

