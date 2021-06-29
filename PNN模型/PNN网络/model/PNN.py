# -*- coding: utf-8 -*-
# @Time    : 2021/3/26 下午3:52
# @Author  : gavin
# @FileName: PNN.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from ProductLayer import Productlayer
from torch import nn
from DNN import DNN
import torch
from torch.nn import functional as F


class PNN(nn.Module):

    def __init__(self, feature_info, hidden_units, mode="in", dnn_dropout=0., embed_dim=10, outdim=1):
        """
        :param feature_info: 特征信息（数值特征，稀疏特征，类别特征的维数隐射）
        :param hidden_units: DNN数目
        :param mode: 特征交叉模式
        :param dropout: 失活比例
        :param embed_dim: 嵌入维度
        :param outdim: 网络输出维度
        """
        self.dense_feats, self.spase_feats, self.spase_feats_map = feature_info
        self.filed_nums = len(self.spase_feats)
        self.dense_nums = len(self.dense_feats)
        self.embed_dim = embed_dim
        self.mode = mode

        # Embedding层
        self.embed_layer = nn.ModuleDict({"embed" + str(key):nn.Embedding(num_embeddings=val, embedding_dim=self.embed_dim)
                                          for key, val in self.spase_feats_map.items()}
                                         )
        # ProductLayer层
        self.produnct = Productlayer(mode, self.embed_dim, self.filed_nums, hidden_units)

        # DNN层
        # 拼接数值特征
        hidden_units[0] += self.dense_nums
        self.dnn_layer = DNN(hidden_units, dnn_dropout)
        self.final_layer =  nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_inputs, spase_inputs = x[:, :13], x[:, 13:]
        spase_inputs = spase_inputs.long()
        # 获取稀疏特征的Embedding向量
        spase_embeds = [self.embed_layer["embed" + key](spase_inputs[:, i]) for key, i in zip(self.spase_feats_map.keys(), range(spase_inputs.shape[1]))]
        # [filed_num, B, embed_dim]
        spase_embeds = torch.stack(spase_embeds)
        spase_embeds = spase_embeds.permute((1, 0, 2))  # [B, filed_num, embed_dim]

        # product layer
        spase_inputs = self.produnct(z, spase_embeds)

        # DNN层
        l1 = F.relu(torch.cat([spase_inputs, dense_inputs], axis=-1))
        dnn_x = self.dnn_layer(l1)
        outputs = F.sigmoid(self.final_layer(dnn_x))

        return outputs


if __name__ == "__main__":
    print(Productlayer())