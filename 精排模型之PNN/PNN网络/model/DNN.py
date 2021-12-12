# -*- coding: utf-8 -*-
# @Time    : 2021/3/26 下午3:52
# @Author  : gavin
# @FileName: DNN.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch import nn
from torch.nn import functional as F


# 定义一个全连接层的神经网络
class DNN(nn.Module):

    def __init__(self, hidden_units, dropout=0):
        super().__init__()

        self.dnn = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in zip(hidden_units[:-1], hidden_units[1:])])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for liner in self.dnn:
            x = liner(x)
            x = F.relu(x)
        x = self.dropout(x)

        return x