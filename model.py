import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import densenet121


class VGG13(nn.Module):
    def __init__(self, num_classes=10, init_weights=False, latent=False):
        super(VGG13, self).__init__()
        self.latent = latent

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
        for layer in self.features:
            x = layer(x)
        pooled_output = self.avgpool(x)
        flattened = torch.flatten(pooled_output, 1)
        before_last_fc_output = self.classifier[:-1](flattened)
        final_output = self.classifier[-1](before_last_fc_output)

        if self.latent:
            return [pooled_output, before_last_fc_output, final_output]

        return final_output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class VGG16(nn.Module):
    def __init__(self, num_classes=10, init_weights=False, latent=False):
        super(VGG16, self).__init__()
        self.latent = latent
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
        for layer in self.features:
            x = layer(x)

        if self.latent:
            pooled_output = x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            before_classifier_output = x

            final_output = self.classifier(x)

            return [pooled_output, before_classifier_output, final_output]
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, latent=False):
        super(ResNet, self).__init__()
        self.latent = latent
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = [] 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.latent:
            pooled_output = x

        x = self.layer4(x)
        if self.latent:
            before_classifier_output = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        final_output = self.fc(x)

        if self.latent:
            return [pooled_output, before_classifier_output, final_output]

        return final_output


def resnet18(num_classes=10, latent=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, latent=latent)
    return model


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, latent=False):
        self.latent = latent
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        last_inception_output = x 

        x = self.maxpool3(x)
        x = self.inception4a(x)

        pooled_output = self.avgpool(x)

        x = torch.flatten(pooled_output, 1)
        x = self.dropout(x)
        final_output = self.fc(x)

        if self.latent:
            return last_inception_output, pooled_output, final_output

        return final_output


class SB_CNN(nn.Module):
    def __init__(self, frames=1025, bands=173, f_size=5, channels=1, num_labels=10, latent=False):
        super(SB_CNN, self).__init__()
        self.latent = latent

        self.conv1 = nn.Conv2d(channels, 24, kernel_size=f_size)
        self.pool = nn.MaxPool2d(kernel_size=(4, 2))
        self.conv2 = nn.Conv2d(24, 48, kernel_size=f_size)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=f_size)
        self.num_flat_features = self._get_conv_output((channels, frames, bands))

        self.fc1 = nn.Linear(self.num_flat_features, 64)
        self.fc2 = nn.Linear(64, num_labels)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        features = []  # 初始化特征列表

        x = F.relu(self.pool(self.conv1(x)))
        if self.latent: features.append(x)

        x = F.relu(self.pool(self.conv2(x)))


        x = F.relu(self.conv3(x))

        if self.latent: pooled_output = x 


        x = x.view(-1, self.num_flat_features)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if self.latent: before_classifier_output = x

        x = self.dropout(x)
        x = self.fc2(x)
        final_output = F.log_softmax(x, dim=1)

        if self.latent:
            return [pooled_output, before_classifier_output, final_output]

        return final_output

    def _get_conv_output(self, shape):
        original_latent = self.latent
        self.latent = False 
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        self.latent = original_latent
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes=10, latent=False):
        super(DenseNet, self).__init__()
        self.latent = latent
        self.densenet_base = densenet121(pretrained=False)
        self.densenet_base.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                      bias=False)

        self.features = self.densenet_base.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_ftrs = self.densenet_base.classifier.in_features
        self.classifier = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        features = self.features(x)

        if self.latent:
            pooled_output = self.pool(features)
            before_classifier_output = torch.flatten(pooled_output, 1)
            final_output = self.classifier(before_classifier_output)
            return features, before_classifier_output, final_output
        else:
            out = F.relu(features, inplace=True)
            out = self.pool(out)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out
