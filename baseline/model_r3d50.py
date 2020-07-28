"""Model definition."""

from torch import nn
import torch.nn.functional as F
import torchvision
import torch
from train_options import parser
from torchvision.models.video import r2plus1d_18
from torch.nn.modules import Conv3d
import numpy as np
from ptflops import get_model_complexity_info
# from train_siamese import DEVICES
from backbone.resnet3d import R3d50,R2Plus1d18,R2Plus1d50

KEY_FEATURES = 512
DROPOUT = 0.25

# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BaselineModel(nn.Module):
    def __init__(self, num_segments,
                 base_model='r2plus1d50'):
        super(BaselineModel, self).__init__()
        self.num_segments = num_segments

        print(("""
Initializing model:
    base model:         {}.
    num_segments:       {}.
        """.format(base_model, self.num_segments)))
        self.dropout = nn.Dropout(DROPOUT)
        # self.base_model_channel_3 = R3d50()
        self.base_model_channel_3 = R2Plus1d50()
        self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3
        feature_dim = getattr(self.base_model_channel_3, 'fc_layer').out_features
        self.fc_layer_1 = nn.Linear(feature_dim, KEY_FEATURES)
        self.fc_layer_2 = nn.Linear(KEY_FEATURES, KEY_FEATURES)
        self.clf_layer = nn.Linear(KEY_FEATURES, 2)

    def forward(self, inputs):
        # ( (img1), (img2))
        outputs = []
        for frames in inputs:
            x = self.data_bn_channel_3(frames)
            x = self.base_model_channel_3(x)
            # print(mix_features.shape)
            outputs.append(x)
        x = self.fc_layer_1(torch.abs(outputs[0] - outputs[1]))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.clf_layer(x)
        return outputs, x


def input_constructer(input_res):
    with torch.cuda.device(2):
        im1=torch.randn(size=(1,3,5,224,224))
        return {'inputs':[im1,im1]}

if __name__ == '__main__':
    with torch.cuda.device(2):
        net = BaselineModel(2,5)

        macs, params = get_model_complexity_info(net,input_res=(224,224),input_constructor=input_constructer, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))