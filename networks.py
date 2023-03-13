'''
    Network for several datasets.
'''

import numpy as np
import torch
from torch import nn, optim, hub
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self, input_size=24, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384):
        """
            MNIST: input_size 28, input_channel 1, classes 10, kernel_size 3, filters1 30, filters2 30, fc200
            Fashion-MNIST: the same as mnist
            KATHER: input_size 150, input_channel 3, classes 8, kernel_size 3, filters1 30, filters2 30, fc 200
            CIFAR10: input_size 24, input_channel 3, classes 10, kernel_size 5, filters1 64, filters2 64, fc 384
        """
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.filters2 = filters2
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=filters1, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=padding)
        self.fc1 = nn.Linear(filters2 * input_size * input_size// 9 , fc_size)
        self.fc2 = nn.Linear(fc_size, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.filters2 * self.input_size * self.input_size // 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(self.dropout(x))
        return F.log_softmax(x, dim=1)



class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        # 构建网络的卷积层和池化层，最终输出命名features，原因是通常认为经过这些操作的输出为包含图像空间信息的特征层
        self.features = self._make_layers(cfg['VGG16'])

        # 构建卷积层之后的全连接层以及分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),  # fc1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(True),
            nn.Linear(512, 10),  # fc3，最终cifar10的输出是10类
        )
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def get_model(model_name,device):
    if model_name == 'CNN':
        return ConvNet().to(device)
    elif model_name == 'VGG16':
        return VGG16().to(device)