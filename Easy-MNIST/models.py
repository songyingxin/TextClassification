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
        self.config = config
        self.LR = nn.Linear(config.input_size, config.output_size)
    
    def forward(self, x):
        x = x.reshape(-1, self.config.input_size)
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
        self.config = config
        self.input_layer = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x):
        x = x.reshape(-1, self.config.input_size)
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.hidden_layer(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.conv2 = nn.Conv2d(6, 16, 3)  
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




