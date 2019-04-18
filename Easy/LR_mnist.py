import torch
import torch.nn as nn
import argparse

import models
import datasets

parser = argparse.ArgumentParser(description="Logistic Regression")

# 参数设置
parser.add_argument('--input_size', default=784, type=int, help="输入的 minist 图片的大小")
parser.add_argument('--output_size', default=10, type=int, help='分类标签数，mnist为10')
parser.add_argument('--epoch_num', default=10, type=int, help="epoch 的数目")
parser.add_argument('--batch_size', default=32, type=int, help="一个 batch 的大小")
parser.add_argument('--learning_rate', default=0.001, type=float, help="学习率")

def main(args):
    
    """ 设备准备 """
    train_loader, test_loader = datasets.minist_data(args)  # 数据
    
    model = models.LogisticRegressionMulti(args)  # Logistic 回归模型
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate) # 优化器选择

    """ Train """
    total_step = len(train_loader)
    for epoch in range(args.epoch_num):
        for i, (images, labels) in enumerate(train_loader):
            # 输入reshape为 [bath_size, input_size]
            images = images.reshape(-1, args.input_size)

            # 前向传播过程
            outputs = model(images)  # 输出预测
            loss = criterion(outputs, labels) # 计算损失

            # 反向传播过程
            optimizer.zero_grad()  # 梯度置0
            loss.backward()
            optimizer.step()

            # 需要观察的信息。 每100个观察一次
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, args.epoch_num, i+1, total_step, loss.item()))
    
    """ Test """
    with torch.no_grad():
        correct = 0
        total = 0
        for (images, labels) in test_loader:
            images = images.reshape(-1, args.input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the model on the 10000 test images: {} %'.format(
            100 * correct / total))

    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == "__main__":
    main(parser.parse_args())
