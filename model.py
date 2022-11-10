import torch.nn as nn
import torch

# ================================================================= 第一种 ： 卷积网络 =================================================================
# 带归一化和激活函数的卷积操作
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out



# 类似LENET-5结构，参考C3D网络
class C3D_Simple(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D_Simple, self).__init__()
        self.conv_1 = BasicConv3d(in_channels=1, out_channels=64, kernel_size=(1,3,3))
        self.pool_1 = nn.AvgPool3d((1,2,2), (1,2,2))
        self.conv_2 = BasicConv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3))
        self.pool_2 = nn.AvgPool3d((1,2,2), (1,2,2))
        self.conv_3a = BasicConv3d(in_channels=128, out_channels=256, kernel_size=3)

        self.pool_3 = nn.AvgPool3d((1,2,2), (1,2,2))
        self.conv_4a = BasicConv3d(in_channels=256, out_channels=512, kernel_size=3)

        self.pool_4 = nn.AvgPool3d(2, 2)
        self.conv_5a = BasicConv3d(in_channels=512, out_channels=256, kernel_size=3)

        self.pool_5 = nn.AdaptiveAvgPool3d(output_size=1)

        self.fc6 = nn.Linear(256, num_classes)
    def forward(self, input):
        x = self.conv_1(input)

        x = self.pool_1(x)

        x = self.conv_2(x)

        x = self.pool_2(x)

        x = self.conv_3a(x)

        x = self.pool_3(x)

        x = self.conv_4a(x)

        x = self.pool_4(x)

        x = self.conv_5a(x)

        x = self.pool_5(x)

        x = x.view(x.shape[0], -1)

        x = self.fc6(x)

        return x




# ================================================================= 第二种 ： 残差网络 =================================================================
#参考U-net网络，增加多个shortcut通道，使用小卷积的叠加操作
class C3D_ResNet_simple(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D_ResNet_simple, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv3d(1, 16, (1, 1, 1), padding=0),
            nn.Conv3d(16, 16, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(16, 16, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(16),

            nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(16, 16, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(16, 16, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(16),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(32, 32, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(32, 32, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(32, 32, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(32),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(64, 64, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1)),
            nn.Conv3d(64, 64, (1, 3, 1), padding=(0, 2, 0), dilation=(1, 2, 1)),
            nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 2), dilation=(1, 1, 2)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (3, 1, 1), padding=(4, 0, 0), dilation=(4, 1, 1)),
            nn.Conv3d(64, 64, (1, 3, 1), padding=(0, 4, 0), dilation=(1, 4, 1)),
            nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 4), dilation=(1, 1, 4)),
            nn.PReLU(64),
        )

        self.layer_4 = nn.Sequential(
            nn.Conv3d(128, 128, (3, 1, 1), padding=(3, 0, 0), dilation=(3, 1, 1)),
            nn.Conv3d(128, 128, (1, 3, 1), padding=(0, 3, 0), dilation=(1, 3, 1)),
            nn.Conv3d(128, 128, (1, 1, 3), padding=(0, 0, 3), dilation=(1, 1, 3)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3, 1, 1), padding=(4, 0, 0), dilation=(4, 1, 1)),
            nn.Conv3d(128, 128, (1, 3, 1), padding=(0, 4, 0), dilation=(1, 4, 1)),
            nn.Conv3d(128, 128, (1, 1, 3), padding=(0, 0, 4), dilation=(1, 1, 4)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3, 1, 1), padding=(5, 0, 0), dilation=(5, 1, 1)),
            nn.Conv3d(128, 128, (1, 3, 1), padding=(0, 5, 0), dilation=(1, 5, 1)),
            nn.Conv3d(128, 128, (1, 1, 3), padding=(0, 0, 5), dilation=(1, 1, 5)),
            nn.PReLU(128),
        )
        self.layer_5 = nn.Sequential(
            nn.Conv3d(128, 256, (3, 1, 1), padding=(1, 0, 0))
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32),
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64),
        )
        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128),
        )
        # self.down_conv4 = nn.Sequential(
        #     nn.Conv3d(128, 256, 2, 2),
        #     nn.PReLU(256)
        # )
        self.drop = nn.Dropout(0.3, True)
        self.map = nn.Sequential(
            nn.Conv3d(128, 64, 1, 1),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, input):
        long_range1 = self.layer_1(input)
        long_range1 += input

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.layer_2(short_range1) + short_range1
        if self.training:
            long_range2 = self.drop(long_range2)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.layer_3(short_range2) + short_range2
        if self.training:
            long_range3 = self.drop(long_range3)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.layer_4(short_range3) + short_range3
        if self.training:
            long_range4 = self.drop(long_range4)

        out = self.map(long_range4)
        out = out.view(out.shape[0], -1)
        out = self.out(out)

        return out



# ================================================================= 第三种 ： 残差网络升级版 =================================================================
#基本3x3尺寸的卷积操作
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# 浅层网络（50层以下）使用的网络基础块 ：BasicBlock
#基本shortcut结构，默认是加上原输入
class BasicBlock(nn.Module):
    expansion = 1 # expansion是对输出通道数的倍乘
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 深层网络（50层及以上）使用的网络基础块 ：Bottleneck
#基本shortcut结构，默认是加上原输入，多一层卷积操作，产生更多的通道
class Bottleneck(nn.Module):
    expansion = 4 # expansion是对输出通道数的倍乘
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#基于ResNet(2D)相似结构，替换网络中所有2D操作为3D操作。
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool_ada = nn.AdaptiveAvgPool3d(output_size=1)

        self.fc = nn.Linear(512, num_classes)
        # self.fc_for_dist = nn.Linear(1, num_classes, bias=False)


        # 设置不同的权值初始化方法
        # 先从self.modules()中遍历每一层，然后判断更曾属于什么类型，是否是Conv2d，是否是BatchNorm2d，是否是Linear的，然后根据不同类型的层，设定不同的权值初始化方法
        # self.modules()是继承torch.nn.Modules()的类拥有的方法，以迭代器形式返回此前声明的所有layers
        for m in self.modules():
            # 对模型卷积层参数以及BN层参数的初始化操作
            # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                # isinstance() 与 type() 区别：
                # type() 不会认为子类是一种父类类型，不考虑继承关系。
                # isinstance() 会认为子类是一种父类类型，考虑继承关系。
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 通过循环创建块来创建层
    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        :param block: 基础块的类型(50层以下)BasicBlock
        :param planes:当前块的输入输入通道数
        :param blocks:块的数目
        :param stride:步长
        :return:
        '''

        # downSample的作用 : 残差连接时,将输入的图像的通道数变成和卷积操作的尺寸一致
        downsample = None
        # 根据ResNet的结构特点，一般只在每层开始时进行判断
        if stride != 1 or self.inplanes != planes * block.expansion: # expansion指的是最后输出的通道数扩充的比例
            # 通道数恢复成一致(块内需要进行相应改变的操作)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        # 创建层
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 返回当前已创建的结构
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool_ada(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out