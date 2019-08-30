"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torch.nn.functional as F
import torchvision
import torch

KEY_FEATURES = 128
DROPOUT = 0


# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation,
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation  # net input, mv,residual,
        self.num_segments = num_segments

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').out_features
        self.dropout = nn.Dropout(DROPOUT)
        self.key_feature_layer = nn.Linear(feature_dim, KEY_FEATURES)
        self.fc_layer_1 = nn.Linear(KEY_FEATURES, KEY_FEATURES)
        self.fc_layer_2 = nn.Linear(KEY_FEATURES, KEY_FEATURES)
        self.clf_layer = nn.Linear(KEY_FEATURES, 2)

        if self._representation == 'mv':
            # here modify the conv1 input channel from resnet 3  ==> 2,that's all
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64,
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)  # input channel is 2
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)  # input channel is 3

    def _prepare_base_model(self, base_model):
        """
        create ResNet backbone
        """
        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            if self._representation == 'iframe':
                self.num_segments = input.shape[1]
            # convert input to 4-dim input
            input = input.view((-1,) + input.size()[-3:])
            if self._representation in ['mv', 'residual']:
                input = self.data_bn(input)

            x = self.base_model(input)
            x = self.dropout(x)
            x = self.key_feature_layer(x)
            x = x.view((-1, self.num_segments) + x.size()[1:])
            x = torch.mean(x, dim=1)
            outputs.append(x)

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
