import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(CRNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.avgpool(x.transpose(1, 2)).squeeze()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor, val)：初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.xavier_uniform_(m.weight)
                # 将所有偏执置为0
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU) and m.bidirectional:
                # 初始化前向GRU的权重
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.xavier_uniform_(param)
                # 初始化反向GRU的权重
                for name, param in m.named_parameters():
                    if 'reverse_weight_ih' in name or 'reverse_weight_hh' in name:
                        nn.init.xavier_uniform_(param)


class VGG13(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(VGG13, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor, val)：初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.xavier_uniform_(m.weight)
                # 将所有偏执置为0
                nn.init.constant_(m.bias, 0)


class VGG16(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor, val)：初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.xavier_uniform_(m.weight)
                # 将所有偏执置为0
                nn.init.constant_(m.bias, 0)
