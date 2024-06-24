import torch
import torch.nn as nn
import torch.optim as optim

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.downsample = downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = x + identity 
        x = self.relu(x)
        
        return x 
    
class ResNet(nn.Module):
    def __init__(self, layers, img_channels, img_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers 
        self.layer1 = self.make_layer(layers[0], 64, 1)
        self.layer2 = self.make_layer(layers[1], 128, 2)
        self.layer3 = self.make_layer(layers[2], 256, 2)
        self.layer4 = self.make_layer(layers[3], 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc = nn.Linear(512*4, img_classes)
        
    def make_layer(self, num_block, out_channels, stride):
        blocks = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(out_channels * 4))
        else:
            downsample = None
            
        blocks.append(Block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4
        
        for i in range(num_block - 1):
            blocks.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x