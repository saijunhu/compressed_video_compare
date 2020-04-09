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

KEY_FEATURES = args.keyfeatures
DROPOUT = args.dropout

QPs = []


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

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)
        # self.qp_model = SimpleConv(1, 64,self.num_segments)
        # self.qp_model.eval()

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model_channel_3, 'fc').out_features
        self.dropout = nn.Dropout(DROPOUT)
        self.key_feature_layer = nn.Linear(feature_dim, KEY_FEATURES)
        self.fc_layer_1 = nn.Linear(KEY_FEATURES * 2, KEY_FEATURES * 2)
        self.fc_layer_2 = nn.Linear(KEY_FEATURES * 2, KEY_FEATURES * 2)
        self.clf_layer = nn.Linear(KEY_FEATURES * 2, 2)

    def _prepare_base_model(self, base_model):
        """
        create 3d convnet backbone
        """
        self.base_model_channel_3 = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.base_model_channel_2 = torchvision.models.video.r2plus1d_18(pretrained=True)
        # here modify the conv1 input channel from resnet 3  ==> 2,that's all
        self.base_model_channel_2.stem[0] = Conv3d(2, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                                   bias=True)

        self.data_bn_channel_2 = nn.BatchNorm3d(2)  # input channel is 2
        self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3

    def forward(self, inputs):
        # ( (img1,mv1,qp1), (img2,mv2,qp2))
        outputs = []
        for features in inputs:
            mix_features = []
            # print(next(self.qp_model.parameters()).is_cuda)
            for i in range(len(features)):
                if i == 0:
                    # for rgb
                    features[i] = self.data_bn_channel_3(features[i])
                    x = self.base_model_channel_3(features[i])
                if i == 1:
                    # for mv and residual need batch_normalization
                    features[i] = self.data_bn_channel_2(features[i])
                    x = self.base_model_channel_2(features[i])
                if i == 2:
                    continue
                x = self.dropout(x)
                x = self.key_feature_layer(x)
                # x = (batch, features)
                mix_features.append(x)
            mix_features = torch.cat([mix_features[0], mix_features[1]], dim=1)
            # print(mix_features.shape)
            outputs.append(mix_features)
        x = self.fc_layer_1(torch.abs(outputs[0] - outputs[1]))
        x = F.relu(x)
        x = self.fc_layer_2(x)
        x = torch.sigmoid(x)
        x = self.clf_layer(x)
        return outputs, x



    @property  # turn a func to a properity
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
