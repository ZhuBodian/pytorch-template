import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class AlexModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 原文的channel数是96，由于相对于原文数据集较小，且为了加快训练速度，这里只取了原文的一半
            # inplace=True允许pytorch通过一种方法增加计算量，降低内存占用
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),  # input[3, 224, 224]  output[48, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),  # output[128, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),  # output[192, 13, 13]
            nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),  # output[192, 13, 13]
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  nn.ReLU(inplace=True),  # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # pytorch的通道排列顺序是batch*channel*height*width，0维是batch是不动的，从channel展
        x = self.classifier(x)
        return x