import argparse
import torch
import torch.nn as nn


import datasets
import models

parser = argparse.ArgumentParser(description="Minist 数据集")

parser.add_argument('--model_name', default="FNN", type=str, help="the model name: FNN, CNN, RNN")

# 训练参数设置， 不同网络参数不同，这里不做细究， 统一共同参数
parser.add_argument('--input_size', default=784,
                    type=int, help="输入的 minist 图片的大小")
parser.add_argument('--output_size', default=10,
                    type=int, help='分类标签数，mnist为10')
parser.add_argument('--epoch_num', default=10, type=int, help="epoch 的数目")
parser.add_argument('--batch_size', default=128, type=int, help="一个 batch 的大小")
parser.add_argument('--learning_rate', default=0.001, type=float, help="学习率")

# 前馈神经网络专属参数
parser.add_argument('--hidden_size', default=500, type=int, help="隐层单元数")



def train(model, optimizer, criterion, train_loader, input_size, epoch, device):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 输入reshape为 [bath_size, input_size]
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播过程
        outputs = model(images)  # 输出预测
        loss = criterion(outputs, labels)  # 计算损失

        # 反向传播过程
        optimizer.zero_grad()  # 梯度置0
        loss.backward()
        optimizer.step()

        # 需要观察的信息。 每100个观察一次
        if(batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, input_size, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
    test_loss /= len(test_loader)  
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(config):

    """ 设备: cpu or GPU """
    print("the current model is {}".format(config.model_name))
    if config.model_name == "LR":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        if torch.cuda.is_available():
            print("device is cuda, # cuda is: ", n_gpu)
        else:
            print("device is cpu")

    """ 模型准备 """
    train_loader, test_loader = datasets.minist_data(config) # 数据
    
    if config.model_name == 'FNN':
        model = models.FNN(config).to(device) # 模型
    elif config.model_name == "LR":
        model = models.LogisticRegressionMulti(config).to(device)
    elif config.model_name == "CNN":
        model = models.CNN().to(device)

        
    criterion = nn.CrossEntropyLoss()  # 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # 优化算法

    """ Train  """
    for epoch in range(1, config.epoch_num + 1):
        train(model, optimizer, criterion, train_loader, config.input_size, epoch, device)
        test(model, test_loader, config.input_size, criterion, device)

    torch.save(model.state_dict(), 'model.ckpt')



if __name__ == "__main__":
    main(parser.parse_args())
