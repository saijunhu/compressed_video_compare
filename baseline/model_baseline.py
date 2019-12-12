"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torch.nn.functional as F
import torchvision
import torch
from train_options import parser
from torchvision.models.video import r2plus1d_18
from torch.nn.modules import Conv3d
import numpy as np

# from train_siamese import DEVICES
args = parser.parse_args()

KEY_FEATURES = 128
DROPOUT = 0.25

QPs = []


# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BaselineModel(nn.Module):
    def __init__(self, num_class, num_segments, representation,
                 base_model='r2plus1d_18'):
        super(BaselineModel, self).__init__()
        self._representation = representation  # net input, mv,residual,
        self.num_segments = num_segments

        print(("""
Initializing model:
    base model:         {}.
    num_segments:       {}.
        """.format(base_model, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)


    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model_channel_3, 'fc').out_features
        self.dropout = nn.Dropout(DROPOUT)
        self.key_feature_layer = nn.Linear(feature_dim, KEY_FEATURES)
        self.fc_layer_1 = nn.Linear(KEY_FEATURES, KEY_FEATURES)
        self.fc_layer_2 = nn.Linear(KEY_FEATURES, KEY_FEATURES)
        self.clf_layer = nn.Linear(KEY_FEATURES, 2)

    def _prepare_base_model(self, base_model):
        """
        create 3d convnet backbone
        """
        self.base_model_channel_3 = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3

    def forward(self, inputs):
        # ( (img1), (img2))
        outputs = []
        for frames in inputs:
            x = self.data_bn_channel_3(frames)
            x = self.base_model_channel_3(x)
            x = self.key_feature_layer(x)
            # print(mix_features.shape)
            outputs.append(x)
        x = self.fc_layer_1(torch.abs(outputs[0] - outputs[1]))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_layer_2(x)
        x = torch.sigmoid(x)
        x = self.clf_layer(x)
        return outputs, x
