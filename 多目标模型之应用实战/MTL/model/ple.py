# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 上午9:34
# @Author  : gavin
# @FileName: ple.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
from torch import nn


class Tower(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, drouout=0.4):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Expert_shared(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Expert_shared, self).__init__()

        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.fc1(x)


class Expert_task1(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Expert_task1, self).__init__()

        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.fc1(x)


class Expert_task2(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Expert_task2, self).__init__()

        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.fc1(x)


class Gate_shared(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Gate_shared, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.fc1(x)


class Gate_task1(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Gate_task1, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.fc1(x)


class Gate_task2(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Gate_task2, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.fc1(x)


class GatingNetwork(nn.Module):

    def __init__(self, input_units, units, num_experts, selectors):
        super(GatingNetwork, self).__init__()

        self.experts_shared = nn.ModuleList([Expert_shared(input_units, units)
                                             for i in range(num_experts)])
        self.experts_task1 = nn.ModuleList([Expert_task1(input_units, units)
                                            for i in range(num_experts)])
        self.experts_task2 = nn.ModuleList([Expert_task2(input_units, units)
                                            for i in range(num_experts)])
        self.expert_activation = nn.ReLU()

        self.gate_shared = Gate_shared(input_units, num_experts * 3)
        self.gate_task1 = Gate_task1(input_units, selectors * num_experts)
        self.gate_task2 = Gate_task2(input_units, selectors * num_experts)

        self.gate_activation = nn.Softmax(dim=-1)
        self.units = units
        self.num_expers = num_experts

    def forward(self, gate_output_shared_final, gate_output_task1_final, gate_output_task2_final):
        # expert shared
        expert_shared_o = [e(gate_output_shared_final)
                           for e in self.experts_shared]
        expert_shared_tensors = torch.cat(expert_shared_o, dim=0)
        expert_shared_tensors = expert_shared_tensors.view(
            -1, self.num_expers, self.units)
        expert_shared_tensors = self.expert_activation(expert_shared_tensors)
        # expert task1
        expert_task1_o = [e(gate_output_task1_final)
                          for e in self.experts_task1]
        expert_task1_tensors = torch.cat(expert_task1_o, dim=0)
        expert_task1_tensors = expert_task1_tensors.view(
            -1, self.num_expers, self.units)
        expert_task1_tensors = self.expert_activation(expert_task1_tensors)
        # expert task2
        expert_task2_o = [e(gate_output_task2_final)
                          for e in self.experts_task2]
        expert_task2_tensors = torch.cat(expert_task2_o, dim=0)
        expert_task2_tensors = expert_task2_tensors.view(
            -1, self.num_expers, self.units)
        expert_task2_tensors = self.expert_activation(expert_task2_tensors)

        # gate task1
        gate_output_task1 = self.gate_task1(gate_output_task1_final)
        gate_output_task1 = self.gate_activation(gate_output_task1)

        gate_expert_output1 = torch.cat(
            [expert_shared_tensors, expert_task1_tensors], dim=1)

        gate_output_task1 = torch.einsum(
            'be,beu ->beu', gate_output_task1, gate_expert_output1)
        gate_output_task1 = gate_output_task1.sum(dim=1)
        # gate task2
        gate_output_task2 = self.gate_task2(gate_output_task2_final)
        gate_output_task2 = self.gate_activation(gate_output_task2)

        gate_expert_output2 = torch.cat(
            [expert_shared_tensors, expert_task2_tensors], dim=1)

        gate_output_task2 = torch.einsum(
            'be,beu ->beu', gate_output_task2, gate_expert_output2)
        gate_output_task2 = gate_output_task2.sum(dim=1)
        # gate shared
        gate_output_shared = self.gate_shared(gate_output_shared_final)
        gate_output_shared = self.gate_activation(gate_output_shared)

        gate_expert_output_shared = torch.cat(
            [expert_task1_tensors, expert_shared_tensors, expert_task2_tensors], dim=1)

        gate_output_shared = torch.einsum(
            'be,beu ->beu', gate_output_shared, gate_expert_output_shared)
        gate_output_shared = gate_output_shared.sum(dim=1)

        return gate_output_shared, gate_output_task1, gate_output_task2


class PLE(nn.Module):

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, hidden_out_size=128, num_experts=2,
                 selectors=2):
        super(PLE, self).__init__()
        if user_feature_dict is None or item_feature_dict is None:
            Exception("用户特征和物品特征不能为空！")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict):
            Exception("输入数据类型必须为字典类型！")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict

        # 共享Embedding(Share bottom)
        user_cate_feature_nums, item_cate_feature_nums = 0, 0

        # 用户特征Embedding编码
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

        # 构建独立任务（tower）
        # Spase feat + Dense feat
        input_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) \
                     + (len(self.user_feature_dict) - user_cate_feature_nums) \
                     + (len(self.item_feature_dict) - item_cate_feature_nums)
        # 实例Multi Layer
        self.gate1 = GatingNetwork(
            input_size, hidden_out_size, num_experts, selectors)

        self.gate2 = GatingNetwork(
            hidden_out_size, hidden_out_size, num_experts, selectors)

        # 实例Tower
        self.towers = nn.ModuleList(
            [Tower(hidden_out_size, 1, 64) for _ in range(num_experts)])

    def forward(self, x):
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))
        # 拼接向量
        user_embed = torch.cat(user_embed_list, dim=1)
        item_embed = torch.cat(item_embed_list, dim=1)
        # hidden_input
        hidden = torch.cat([user_embed, item_embed], axis=1).float()

        gate_output_shared, gate_output_task1, gate_output_task2 = self.gate1(
            hidden,hidden, hidden)
        _, task1_o, task2_o = self.gate2(
            gate_output_shared, gate_output_task1, gate_output_task2)

        final_output = [tower(task) for tower, task in zip(
            self.towers, [task1_o, task2_o])]

        return final_output


if __name__ == "__main__":
    import numpy as np

    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))

    user_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (1, 4)}
    item_cate_dict = {'item_id': (8, 1), 'item_cate': (6, 2), 'item_num': (1, 5)}
    ple = PLE(user_cate_dict, item_cate_dict)
    print(ple(a))

