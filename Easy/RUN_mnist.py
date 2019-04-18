import argparse
import torch
import torch.nn as nn


import datasets
import models

parser = argparse.ArgumentParser(description="Logistic Regression")

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

def main(config):

    """ 设备准备 """
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
    criterion = nn.CrossEntropyLoss()  # 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # 优化算法

    """ Train  """
    total_step = len(train_loader)
    for epoch in range(config.epoch_num):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, config.input_size).to(device)
            labels = labels.to(device)

            # 前向传播 
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config.epoch_num, i+1, total_step, loss.item()))

    """ Test """
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, config.input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test images: {} %'.format(
            100 * correct / total))

    torch.save(model.state_dict(), 'model.ckpt')



if __name__ == "__main__":
    main(parser.parse_args())
