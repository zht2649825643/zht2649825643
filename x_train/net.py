# -*- coding: utf-8 -*
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)   # 224
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                         # 112
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                         # 56
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)                         # 28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.relu4 = nn.ReLU(True)
        self.fc2 = nn.Linear(256, 4)
        # self.softmax = nn.Softmax(True)

    def forward(self, in_data):
        output = self.conv1(in_data)
        output = self.relu1(output)
        output = self.bn1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.bn2(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = self.relu3(output)
        output = self.bn3(output)
        output = self.pool3(output)

        output = output.view(-1, 128 * 28 * 28)
        output = self.fc1(output)
        output = self.relu4(output)
        output = self.fc2(output)
        # print("===1", output)
        output = output.softmax(dim=1)
        # print("===2", output)

        return output


class ALexNet(nn.Module):
    def __init__(self):
        super(ALexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),  # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 由于使用CPU镜像，精简网络，若为GPU镜像可添加该层
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output




