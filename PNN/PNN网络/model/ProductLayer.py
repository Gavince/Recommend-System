# -*- coding: utf-8 -*-
# @Time    : 2021/3/26 下午3:53
# @Author  : gavin
# @FileName: ProductLayer.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch import nn
import torch


class Productlayer(nn.Module):

    def __init__(self, mode, embed_nums, filed_nums, hidden_units):
        """
        :param mode: l_P互操作方式
        :param embed_nums: Embedding嵌入维度
        :param filed_nums:
        :param hidden_units:
        """
        super(Productlayer, self).__init__()
        self.mode = mode
        # z部分
        self.w_z = nn.Parameter(torch.rand([filed_nums, embed_nums, hidden_units[0]]))

        # p部分
        if self.mode == "in":  # 内积方式
            self.w_p = nn.Parameter(torch.rand([filed_nums, filed_nums, hidden_units[0]]))
        else:  # 外积方式
            self.w_p = nn.Parameter(torch.rand([embed_nums, embed_nums, hidden_units[0]]))

        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)

    def forward(self, z, sparse_embeds):

        #  l_z:线性部分
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # [B, hidden_units[0]]

        #  l_p:特征交叉部分
        if self.mode == "in":
            p = torch.matmul(sparse_embeds, sparse_embeds((0, 2, 1)))  # [B, field_dim, field_dim]
        else:
            f_sum = torch.unsqueeze(torch.sum(sparse_embeds, dim=1), dim=1)  # [B, 1, embed_dim]
            p = torch.matmul(f_sum.permute((0, 2, 1)), f_sum)  # [B, embed_dim]

        l_p = torch.mm(p.reshape(p.shape[0], -1), self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)

        output = l_p + l_z + self.l_b

        return output