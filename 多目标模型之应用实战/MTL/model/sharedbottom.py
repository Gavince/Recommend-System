# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 下午9:41
# @Author  : gavin
# @FileName: sharedbottom.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn
from layer import DNN


class SharedBottom(nn.Module):

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, activation="relu",
                 bottom_hidden_size=[256, 128], tower_hidden_size=[128, 64], num_tasks=2, tasks_name=["ctr", "cvr"],
                 use_bn=False, dropout_rate=0, seed=1024):
        """

        :param user_feature_dict:
        :param item_feature_dict:
        :param emb_dim:
        :param activation:
        :param bottom_hidden_size:
        :param tower_hidden_size:
        :param num_tasks:
        :param tasks_num:
        :param use_bn:
        :param dropout_rate:
        :param seed:
        """

        super(SharedBottom, self).__init__()
        if user_feature_dict is None or item_feature_dict is None:
            Exception("用户特征和物品特征不能为空！")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict):
            Exception("输入数据类型必须为字典类型！")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_tasks = num_tasks
        self.tasks_name = tasks_name

        # 构建Embedding输入
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            # 必须为Spase Feature
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        # 物品特征
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # Spase feat + Dense feat
        input_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) \
                     + (len(self.user_feature_dict) - user_cate_feature_nums) \
                     + (len(self.item_feature_dict) - item_cate_feature_nums)
        # 共享层
        self.shared_bottom_layer = DNN(input_dim=input_size, hidden_units=bottom_hidden_size,
                                       activation=activation
                                       , use_bn=use_bn, dropout_rate=dropout_rate)
        # 子任务层
        for i in range(num_tasks):
            setattr(self, "tower_{}_dnn".format(tasks_name[i]), nn.Sequential(DNN(input_dim=bottom_hidden_size[-1],
                                                                                     hidden_units=tower_hidden_size,
                                                                                     activation=activation,
                                                                                     use_bn=use_bn,
                                                                                     dropout_rate=dropout_rate)
                                                                                 ,
                                                                                 nn.Linear(tower_hidden_size[-1], 1)))

    def forward(self, x):
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(
                    getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(
                    getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))
        # 拼接向量
        user_embed = torch.cat(user_embed_list, dim=1)
        item_embed = torch.cat(item_embed_list, dim=1)
        dnn_input = torch.cat([user_embed, item_embed], axis=1).float()
        # bottom_layer
        shared_bottom_output = self.shared_bottom_layer(dnn_input)
        # tower_layer
        task_outputs = []
        for i in range(self.num_tasks):
            net = getattr(self, "tower_{}_dnn".format(self.tasks_name[i]))
            dnn_output = net(shared_bottom_output)
            task_outputs.append(dnn_output)

        return task_outputs


if __name__ == "__main__":
    import numpy as np

    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))

    user_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (1, 4)}
    item_cate_dict = {'item_id': (8, 1), 'item_cate': (6, 2), 'item_num': (1, 5)}
    sharebottom = SharedBottom(user_cate_dict, item_cate_dict)
    print(sharebottom(a))
    print(sharebottom)
