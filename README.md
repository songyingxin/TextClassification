

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

## Easy

```
python RUN_mnist.py --model_name=LR or CNN or RNN, or FNN
```

| Dataset            | LR   | FNN  | CNN  | RNN  |
| ------------------ | ---- | ---- | ---- | ---- |
| Mnist 手写数字识别 | 92%  | 98%  | 99%  |      |




## Medium-CV

- 数据集：SST-2

## Medium-NLP

- 数据集：MR


