"""Model definition."""

from torch import nn
import torch.nn.functional as F
import torchvision
import torch
from train_options import parser
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.modules import Conv3d
import numpy as np

# from train_siamese import DEVICES
args = parser.parse_args()

KEY_FEATURES = 128
DROPOUT = args.dropout


# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleConv(nn.Module):
    def __init__(self, input_planes, output_planes, num_segments):
        super(SimpleConv, self).__init__()
        self.num_segments = num_segments
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_planes, output_planes, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # 2,10,1,112,112
        x = x.view((-1,) + x.size()[-3:])
        x = self.layer1(x)
        x = x.view((-1, self.num_segments,) + x.size()[-3:])
        return x


class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation,
                 base_model='r2plus1d_18'):
        super(Model, self).__init__()
        self._representation = representation  # net input, mv,residual,
        self.num_segments = num_segments

        print(("""
Initializing model:
    base model:         {}.
    num_segments:       {}.
        """.format(base_model, self.num_segments)))

        self.data_bn_channel_2 = nn.BatchNorm3d(2)  # input channel is 2
        self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3
        self.base_model_channel_3 = VideoResNetAttention3C()
        self.base_model_channel_2 = VideoResNetAttention2C()

        base_out = getattr(self.base_model_channel_3, 'fc_layer').out_features # preset 512
        self.key_feature_layer = nn.Linear(base_out, KEY_FEATURES)
        self.fc_layer_1 = nn.Linear(KEY_FEATURES * 2, KEY_FEATURES * 2)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc_layer_2 = nn.Linear(KEY_FEATURES * 2, KEY_FEATURES * 2)
        self.clf_layer = nn.Linear(KEY_FEATURES * 2, 2)



    def forward(self, inputs):
        # ( (img1,mv1,qp1), (img2,mv2,qp2))
        outputs = []
        for features in inputs:
            mix_features = []
            for i in range(len(features)):
                if i == 0:
                    # for rgb,qp
                    features[i] = self.data_bn_channel_3(features[i])
                    x = self.base_model_channel_3((features[i], features[i + 2]))
                if i == 1:
                    # for mv and residual need batch_normalization
                    features[i] = self.data_bn_channel_2(features[i])
                    x = self.base_model_channel_2(features[i])
                if i == 2:
                    continue
                x = self.key_feature_layer(x)
                # x = (batch, features)
                mix_features.append(x)
            mix_features = torch.cat([mix_features[0], mix_features[1]], dim=1)
            # print(mix_features.shape)
            outputs.append(mix_features)
        x = self.fc_layer_1(torch.abs(outputs[0] - outputs[1]))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.clf_layer(x)
        return outputs, x


#######################################################################################
#######################################################################################
#######################################################################################
## the following code from src and modified

class Simple3dConv(nn.Module):
    def __init__(self, input_planes, output_planes):
        super(Simple3dConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(input_planes, 32, kernel_size=3, padding=1, stride=1),
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, output_planes, kernel_size=3, padding=1, stride=1),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_planes),
            nn.ReLU(inplace=True)
        )
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        out = self.layer2(x)
        return out


model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class VideoResNetAttention3C(nn.Module):
    def __init__(self, arch='r2plus1d_18', pretrained=True, progress=True,
                 block=BasicBlock,
                 conv_makers=[Conv2Plus1D] * 4,
                 layers=[2, 2, 2, 2],
                 stem=R2Plus1dStem, keyfeatures=512,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNetAttention3C, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.attention_layer = Simple3dConv(1, 64)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_layer = nn.Linear(512 * block.expansion, keyfeatures)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        if pretrained:
            pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                       progress=progress)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, inputs):
        # assert x= (img1,qp)
        x, y = inputs
        y.transpose_(1,2)
        x = self.stem(x)
        x = self.layer1(x)
        ##open qp or not
        y = self.attention_layer(y)
        x = x.mul(y)
        ###
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc_layer(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VideoResNetAttention2C(nn.Module):

    def __init__(self, arch='r2plus1d_18', pretrained=True, progress=True,
                 block=BasicBlock,
                 conv_makers=[Conv2Plus1D] * 4,
                 layers=[2, 2, 2, 2],
                 stem=R2Plus1dStem, keyfeatures=512,
                 zero_init_residual=False):
        """Generic resnet video generator.

            Args:
                block (nn.Module): resnet building block
                conv_makers (list(functions)): generator function for each layer
                layers (List[int]): number of blocks per layer
                stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
                num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
                zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
            """

        super(VideoResNetAttention2C, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        # self.attention_layer = Simple3dConv(1, 64)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_layer = nn.Linear(512 * block.expansion, keyfeatures)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        if pretrained:
            pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                       progress=progress)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict, strict=False)
        # change channel to 2c
        # modify channel sjhu
        self.stem[0] = Conv3d(2, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                              bias=True)

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
