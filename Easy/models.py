import torch
import torch.nn as nn
import torch.nn.functional as F

# Logistic Regression 二分类 
class LogisticRegressionBinary(nn.Module):
    def __init__(self, config):
        super(LogicRegression, self).__init__()
        self.LR = nn.Linear(config.input_size, config.output_size)
    
    def forward(self, x):
        out = self.LR(x)
        out = torch.sigmoid(out)
        return out


class LogisticRegressionMulti(nn.Module):
    def __init__(self, config):
        super(LogisticRegressionMulti, self).__init__()
        self.LR = nn.Linear(config.input_size, config.output_size)
    
    def forward(self, x):
        return self.LR(x)


class LinearRegression(nn.Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.LR = nn.Linear(config.input_size, config.output_size)

    def forward(self, x):
        out = self.LR(x)
        return out

# 前馈神经网络, 一个隐层
class FNN(nn.Module):
    def __init__(self, config):
        super(FNN, self).__init__()
        self.input_layer = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.hidden_layer(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out



class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        


