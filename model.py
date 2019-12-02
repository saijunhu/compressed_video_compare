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


# class SimpleConv(nn.Module):
#     def __init__(self, input_planes, output_planes, num_segments):
#         super(SimpleConv, self).__init__()
#         self.num_segments = num_segments
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(input_planes, output_planes, kernel_size=3, padding=1, stride=1),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         )
#
#     def forward(self, x):
#         # 2,10,1,112,112
#         # x = x.view((-1,) + x.size()[-3:])
#         # x = self.layer1(x)
#         # x = x.view((-1, self.num_segments,) + x.size()[-3:])
#
#         return a


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
            global QPs
            QPs = self.deal_qp_data(features[2])
            # print(next(self.qp_model.parameters()).is_cuda)
            for i in range(len(features)):
                if i == 0:
                    # for rgb
                    features[i] = self.data_bn_channel_3(features[i])
                    handle = self.base_model_channel_3.layer1.register_forward_hook(layer1_hook_fn_forward)
                    x = self.base_model_channel_3(features[i])
                    handle.remove()
                if i == 1:
                    # for mv and residual need batch_normalization
                    features[i] = self.data_bn_channel_2(features[i])
                    handle = self.base_model_channel_2.layer1.register_forward_hook(layer1_hook_fn_forward)
                    x = self.base_model_channel_2(features[i])
                    handle.remove()
                if i == 2:
                    continue
                x = self.dropout(x)
                x = self.key_feature_layer(x)
                # x = (batch, features)
                mix_features.append(x)
            mix_features = torch.cat([mix_features[0], QPs.mean().item()*mix_features[1]], dim=1)
            # print(mix_features.shape)
            outputs.append(mix_features)
        x = self.fc_layer_1(torch.abs(outputs[0] - outputs[1]))
        x = F.relu(x)
        x = self.fc_layer_2(x)
        x = torch.sigmoid(x)
        x = self.clf_layer(x)
        return outputs, x

    def deal_qp_data(self, x):
        # print(qps.shape)
        x.transpose_(1,2)
        a = x.repeat((1,64,1,1,1))
        return a

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


def layer1_hook_fn_forward(module, input, output):
    # output shape 2, 64, 10, 56, 56
    global QPs
    a = QPs.to(output.device)
    assert a.device == output.device, "tensor device not match"
    return output.mul(a)
