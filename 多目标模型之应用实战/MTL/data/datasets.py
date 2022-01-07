# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 下午4:34
# @Author  : gavin
# @FileName: datasets.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
from torch.utils import data


class TrainDateSet(data.Dataset):

    def __init__(self, data):
        self.features = data.iloc[:, :-2].values
        self.label1 = data.iloc[:, -2].values
        self.label2 = data.iloc[:, -1].values

    def __getitem__(self, index):
        return self.features[index], self.label1[index], self.label2[index]

    def __len__(self):
        return len(self.features)
