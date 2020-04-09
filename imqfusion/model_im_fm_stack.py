"""Model definition."""

from torch import nn
import torch.nn.functional as F
import torchvision
import torch
from train_options import parser
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.modules import Conv3d
import numpy as np
from backbone.resnet3d import R2Plus1d
# from train_siamese import DEVICES
args = parser.parse_args()

KEY_FEATURES = 128
DROPOUT = args.dropout


class IframeNet(R2Plus1d):
    def __init__(self):
        super(IframeNet,self).__init__(num_classes=128)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class MvNet(R2Plus1d):
    def __init__(self):
        super(MvNet,self).__init__(input_channels=2,num_classes=128)


    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



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

        self.data_bn_channel_1 = nn.BatchNorm3d(1)  # input channel is 2
        self.data_bn_channel_2 = nn.BatchNorm3d(2)  # input channel is 2
        self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3
        self.mvnet = MvNet()
        self.iframenet = IframeNet()
        self.fusion = Simple3dConv(1024,1024)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc_layer = nn.Linear(1024, 2)



    def forward(self, inputs):
        # ( (img1,mv1,qp1), (img2,mv2,qp2))
        outputs = []
        for features in inputs:
            mix_features = []
            for i in range(len(features)):
                if i == 0:
                    # for rgb,qp
                    features[i] = self.data_bn_channel_3(features[i])
                    x = self.iframenet(features[i])
                if i == 1:
                    # for mv and residual need batch_normalization
                    features[i] = self.data_bn_channel_2(features[i])
                    x = self.mvnet(features[i])
                if i == 2:
                    continue
                    features[i] = self.data_bn_channel_1(features[i])
                    x = self.qpnet(features[i])
                # x = (batch, features)
                mix_features.append(x)
            mix_features = torch.cat((mix_features[0],mix_features[1]),dim=1)#TODO
            mix_features = self.fusion(mix_features)
            mix_features = self.avgpool(mix_features)
            mix_features = mix_features.flatten(1)
            # print(mix_features.shape)
            outputs.append(mix_features)
        x = self.fc_layer(torch.abs(outputs[0] - outputs[1]))
        return outputs, x


#######################################################################################
#######################################################################################
#######################################################################################
## the following code from src and modified

class Simple3dConv(nn.Module):
    def __init__(self, input_planes, output_planes):
        super(Simple3dConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(input_planes, output_planes, kernel_size=1, padding=1, stride=1),
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
        # out = self.layer2(x)
        return x

