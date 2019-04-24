
## Introduction

该仓库记录了我从 TensorFlow 转向 Pytorch 时做的小项目， 主要针对 NLP 初学者，分为三个难度：

- Easy 级别简单实现了一下常见基础模型：逻辑回归， 线性回归， 前馈神经网络， 卷积神经网络，循环神经网络；
- Medium 级别针对NLP 初学者，采用最简单的文本分类任务， 实现了一些经典模型，如TextCNN， FastText等； 
- Hard 模型以 Squad 数据集为基础，实现了一些阅读理解复杂模型， 理解阅读理解领域对你的学习是很有帮助的。

## Requirement

- python3
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

数据集源自 [glue](https://gluebenchmark.com/tasks)， 考虑到 sst-2 并没有给出测试集，因此我将训练集的2000多个样本划分出来形成测试集， 划分代码见：split_train.py 。



### BertBaseline

本次实验是在单1080ti上运行，为了精简代码，易于理解，去掉了一些无关参数和逻辑， 毕竟原仓库中的代码1000多行，看起来不要太累。

## Results

```
python run_TextCNN.py / run_TextRNN.py
```

Dataset | TextCNN | TextRNN | FastText
--- | --- | --- | ---
SST-2 | 92.97 | 92.38 |


- 待做： 采用 highway networks 将词向量与char-level向量结合起来