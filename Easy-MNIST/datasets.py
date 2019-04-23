import torch
import torchvision
import torchvision.transforms as transforms

root_dir = "../../data"

def minist_data(config):
    """ 导入 Minist 数据集 """
    train_dataset = torchvision.datasets.MNIST(
        root=root_dir, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=root_dir, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader
