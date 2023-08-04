import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init
import math
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
#from PIL import ImageFile
from PIL import Image
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import base64
from io import BytesIO
import numpy as np
import os
import json
import sys
import cv2

class_labels = ['하트형/역삼각형 얼굴', '긴 얼굴', '계란형 얼굴', '둥근형 얼굴', '네모형/각진얼굴']

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]
        oup_inc = oup // 2

        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along the channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        # else:
        #     raise ValueError(
        #         """{} groups is not supported for
        #         1x1 Grouped Convolutions""".format(num_groups))

        # building the first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


def load_model(checkpoint_path):
    model = ShuffleNetV2(n_class=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(checkpoint_path, map_location=device)['model_state']
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_data):
    image_stream = BytesIO(image_data)
    image_data = Image.open(image_stream)
    image_bytes = image_data.tobytes()
    if image_data is None:
        raise ValueError("Failed to read image")
    
    print('ㅂㅈㄷㅂㅈㄷ')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    print('ㅂㅈㄷㅂㅈㄷ')
    input_tensor = transform(image_data).unsqueeze(0)
    return input_tensor

# def preprocess_image(image_data):
#     try:
#         image_stream = io.BytesIO(image_data)
#         image = Image.open(image_stream)
#         if image is None:
#             raise ValueError("Failed to read image")

#         transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#         input_tensor = transform(image).unsqueeze(0)
#         return input_tensor
#     except Exception as e:
#         print(str(e))
#         raise e

def predict_image(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)
    for i in range(len(prediction)):
        predicted_class_value = prediction[i]
        predicted_class_label = class_labels[i]
        prediction_dict = {
            "class": predicted_class_label,
            "output_softmax": predicted_class_value
        }        
    return prediction

def tensor_to_list(tensor_data):
    # 텐서를 numpy 배열로 변환
    numpy_data = tensor_data.cpu().numpy()  # 만약 GPU에서 텐서가 생성되었다면, 먼저 CPU로 이동시킵니다.

    # numpy 배열을 리스트로 변환
    list_data = numpy_data.tolist()

    return list_data