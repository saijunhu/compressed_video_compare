"""Model definition."""

from torch import nn
import torch.nn.functional as F
import torchvision
import torch
from train_options import parser
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.modules import Conv3d
import numpy as np
from SJUtils.resnet3d import R2Plus1d18
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

        self.data_bn_channel_1 = nn.BatchNorm3d(1)  # input channel is 2
        self.data_bn_channel_2 = nn.BatchNorm3d(2)  # input channel is 2
        self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3
        self.rgbnet = R2Plus1d18(num_classes=128)
        self.mvnet = R2Plus1d18(input_channels=2,num_classes=128)
        self.qpnet = R2Plus1d18(input_channels=1,num_classes=128)

        # base_out = getattr(self.base_model_channel_3, 'fc_layer').out_features # preset 512
        # self.key_feature_layer = nn.Linear(base_out, KEY_FEATURES)
        self.fc_layer = nn.Linear(KEY_FEATURES * 3, KEY_FEATURES * 3)
        self.dropout = nn.Dropout(DROPOUT)
        self.clf_layer = nn.Linear(KEY_FEATURES * 3, 2)



    def forward(self, inputs):
        # ( (img1,mv1,qp1), (img2,mv2,qp2))
        outputs = []
        for features in inputs:
            mix_features = []
            for i in range(len(features)):
                if i == 0:
                    # for rgb,qp
                    features[i] = self.data_bn_channel_3(features[i])
                    x = self.rgbnet(features[i])
                if i == 1:
                    # for mv and residual need batch_normalization
                    features[i] = self.data_bn_channel_2(features[i])
                    x = self.mvnet(features[i])
                if i == 2:
                    features[i] = self.data_bn_channel_1(features[i])
                    x = self.qpnet(features[i])
                # x = (batch, features)
                mix_features.append(x)
            mix_features = torch.cat([mix_features[0], mix_features[1],mix_features[2]], dim=1)
            # print(mix_features.shape)
            outputs.append(mix_features)
        x = self.fc_layer(torch.abs(outputs[0] - outputs[1]))
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

