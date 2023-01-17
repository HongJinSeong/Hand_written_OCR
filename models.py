from transformation import *
from predictions import *
from feature_extraction import *
from collections import namedtuple

from torch.nn.parameter import Parameter
from torch import einsum
import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial
import numpy as np
from einops import rearrange


def compute_loss(criterion, text_batch_logits, text_batch_targets, text_batch_targets_lens, device):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2)  # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device)  # [batch_size]

    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss


## naver ai 참조 deep-text-recognition-benchmark TPS-RES_ATTN ( 3060으로 버거워서 포기  )
class TPS_RES_ATTN(nn.Module):
    def __init__(self):
        super(TPS_RES_ATTN, self).__init__()
        self.Transformation = TPS_SpatialTransformerNetwork(F = 20, I_size=(100, 300), I_r_size=(100, 300), I_channel_num=3)
        self.FeatureExtraction = ResNet_FeatureExtractor(3, 512)

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.bi_lstm1 = BidirectionalLSTM(512, 2500, 2500)
        self.bi_lstm2 = BidirectionalLSTM(2500, 2500, 2500)

        self.Prediction = Attention(2500, 2500, 2351)

    def forward(self, imgs, text, is_train=True):
        input = self.Transformation(imgs)
        x = self.FeatureExtraction(input)
        x = x.permute(0, 3, 1, 2)
        x = self.AdaptiveAvgPool(x)
        x = x.squeeze(3)
        x = self.bi_lstm1(x)
        x = self.bi_lstm2(x)
        x = self.Prediction(x, text, is_train = is_train)
        return x



class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

#### naver ocr git 참조
#### output을 8개로 뽑음
class RCNN_OCR(nn.Module):
    def __init__(self):
        super(RCNN_OCR, self).__init__()
        output_channel = 512
        input_channel = 1

        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2),  # 256x4x25
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            # nn.MaxPool2d((2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True),
            # nn.MaxPool2d((2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], (1,2), 1, 0), nn.ReLU(True))  # 512x1x24

        self.ConvNet2 = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 17, 8, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            # nn.MaxPool2d((2, 1)),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            # nn.MaxPool2d((2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], (1,2), 1, 0), nn.ReLU(True),
            )

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.rnn1 = BidirectionalLSTM(1024, 2500, 2500)

        self.fc = nn.Linear(2500, 2350)

    def forward(self, imgs):
        ## 압축된 feature w가 time step으로 쓰임
        f1 = self.ConvNet(imgs)  # [b, c, h, w]
        f2 = self.ConvNet2(imgs)
        x = torch.cat((f1,f2),dim=1)
        x = x.permute(0, 3, 1, 2)  # ->  [b, w, c, h]
        x = self.AdaptiveAvgPool(x)
        x = x.squeeze(3)

        x = self.rnn1(x)

        x = self.fc(x)  # 최종 output batch x sequence x 2350

        return x

#### CRAFT

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
