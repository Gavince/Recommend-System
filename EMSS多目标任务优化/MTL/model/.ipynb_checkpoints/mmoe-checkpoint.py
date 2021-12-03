# -*- coding: utf-8 -*-
# @Time    : 2021/11/8 下午9:49
# @Author  : gavin
# @FileName: mmoe.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import torch
import torch.nn as nn


class MMoE(nn.Module):

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, n_expert=3, mmoe_hidden_dim=128,
                 hidden_dim=[128, 64], output_size=1, num_tasks=2, expert_activation=None):
        """

        :param user_feature_dict:
        :param item_feature_dict:
        :param emb_dim:
        :param n_expert:
        :param mmoe_hidden_dim:
        :param hidden_dim:
        :param output_size:
        :param num_tasks:
        """
        super(MMoE, self).__init__()

        if user_feature_dict is None or item_feature_dict is None:
            Exception("用户特征和物品特征不能为空！")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict):
            Exception("输入数据类型必须为字典类型！")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_tasks = num_tasks

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
        hidden_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) \
                      + (len(self.user_feature_dict) - user_cate_feature_nums) \
                      + (len(self.item_feature_dict) - item_cate_feature_nums)

        # 专家网络
        self.erperts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.erperts.data.normal_(0, 1)
        self.erperts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)

        # 门控网络
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True)
                      for _ in range(num_tasks)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gate_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_tasks)]

        for i in range(self.num_tasks):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            # input: mmoe_hidden_dim + hidden_dim
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x):

        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # 编码Embedding向量
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
        # B*hidden
        hidden = torch.cat([user_embed, item_embed], dim=1).float()
        # MMoE
        expert_outs = torch.matmul(hidden, self.erperts.permute(1, 0, 2)).permute(1, 0, 2)  # B*mmoe_hidden_dim*experts
        expert_outs += self.erperts_bias
        # 门控单元
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.mm(hidden, gate)  # B * num_experts
            if self.gate_bias:
                gate_out += self.gate_bias[idx]
            # 归一化
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)
        # 各个模块
        outs = list()
        for gate_out in gates_out:
            expand_gate_out = torch.unsqueeze(gate_out, dim=1)  # B * 1 * experts
            weighted_expert_output = expert_outs * expand_gate_out.expand_as(expert_outs)  # B * mmoe_hidden * expert
            outs.append(torch.sum(weighted_expert_output, 2))  # B * mmoe_hidden

        # task_tower
        task_outputs = list()
        for i in range(self.num_tasks):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)

        return task_outputs


if __name__ == "__main__":
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))
    user_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (1, 4)}
    item_cate_dict = {'item_id': (8, 1), 'item_cate': (6, 2), 'item_num': (1, 5)}
    mmoe = MMoE(user_cate_dict, item_cate_dict)
    mmoe.to(device)
    a = a.to(device)
    outs = mmoe(a)
    print(outs)




