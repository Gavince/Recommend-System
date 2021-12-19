# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 下午9:42
# @Author  : gavin
# @FileName: DNN.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281i
import torch
from torch import nn
from .activation import activation_layer


class DNN(nn.Module):

    def __init__(self, input_dim, hidden_units, activation="relu", dice_dim=3, dropout_rate=0, use_bn=False,
                 init_std=0.0001,
                 seed=1024):
        """

        :param input_dim: 输入维度
        :param hidden_units: 隐藏层维度
        :param activation: 激活函数
        :param dice_dim:
        :param dropout_rate: s
        :param use_bn:
        :param init_std:
        :param seed:
        """
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is enmpty!")

        hidden_units = [input_dim] + list(hidden_units)
        # 线性层
        self.linears = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i + 1]
                                                ) for i in range(len(hidden_units) - 1)]
                                     )
        # 1d-BN层
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        # 激活函数
        self.activation_layers = nn.ModuleList([activation_layer(activation, hidden_units[i + 1], dice_dim
                                                                               ) for i in range(len(hidden_units) - 1)]
                                               )

        # 初始化网络参数
        # for name, param in self.linears.named_parameters():
        #     if "weight" in name:
        #         nn.init.normal_(param, mean=0, std=init_std)

    def forward(self, x):
        deep_input = x

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc

        return deep_input


if __name__ == "__main__":
    mlp = DNN(128, [128, 64])
    print(mlp(torch.rand([5, 128])))
