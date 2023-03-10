import torch
import torch.nn as nn
import math


class FRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(FRN, self).__init__()
        tau = torch.randn(1, requires_grad=True)
        beta = torch.randn((1, channels, 1, 1), requires_grad=True)
        gamma = torch.randn((1, channels, 1, 1), requires_grad=True)

        self.tau = nn.Parameter(tau)
        self.beta = nn.Parameter(beta)
        self.gamma = nn.Parameter(gamma)

        self.register_buffer('mytao', self.tau)
        self.register_buffer('mybeta', self.beta)
        self.register_buffer('mygamma', self.gamma)
        self.eps = eps

    def forward(self, x):
        nu2 = torch.mean(torch.pow(x, 2), dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = torch.max((self.gamma * x + self.beta), self.tau)

        return y


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = FRN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = FRN(planes)
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

class ResNet(nn.Module):
    '''
    Basic ResNet remove the classify layers 
    '''

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = FRN(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                FRN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        res1 = self.layer1(x)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        return res1, res2, res3, res4


def Encoder(**kwargs):
    """Constructs a model to get Image Embedding.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


if __name__ == "__main__":
    from torchsummary import summary

    m = Encoder()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m.to(device)
    summary(m, input_size=(3, 512, 512))
