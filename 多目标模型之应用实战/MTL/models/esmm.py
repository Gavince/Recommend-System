import torch
from torch import nn


class ESMM(nn.Module):

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, hidden_dim=[128, 64], dropouts=[0.5, 0.5],
                 output_size=1, task_name=["ctr", "cvr"]):
        """

        :param user_feature_dict: 用户特征
        :param item_feature_dict:　物品特征
        :param emb_dim: 128
        :param hidden_dim: [128, 64]
        :param dropout: 0.5
        :param output_size: 1
        :param num_tasks:2
        """
        super(ESMM, self).__init__()

        if user_feature_dict is None or item_feature_dict is None:
            Exception("用户特征和物品特征不能为空！")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict):
            Exception("输入数据类型必须为字典类型！")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_tasks = len(task_name)
        self.task_name = task_name

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

        for i in range(self.num_tasks):
            setattr(self, 'task_{}_dnn'.format(self.task_name[i]), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(self.task_name[i])).add_module('hidden_{}'.format(j),
                                                                                  nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(self.task_name[i])).add_module('batchnorm_{}'.format(j),
                                                                                  nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, "task_{}_dnn".format(self.task_name[i])).add_module("{}_activation".format(task_name[i])
                                                                                  , nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(self.task_name[i])).add_module('dropout_{}'.format(j),
                                                                                  nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(self.task_name[i])).add_module('task_{}_last_layer'.format(j),
                                                                              nn.Linear(hid_dim[-1], output_size))

    def forward(self, x):
        #         assert x.size()[1] != len(self.item_feature_dict) + len(self.user_feature_dict)
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
        hidden = torch.cat([user_embed, item_embed], axis=1).float()

        # 子网络
        task_outputs = list()
        for i in range(self.num_tasks):
            x = hidden
            # 　Module list
            for mod in getattr(self, 'task_{}_dnn'.format(self.task_name[i])):
                x = mod(x)
            task_outputs.append(x)

        if self.num_tasks == 2:

            pCTCVR = torch.mul(task_outputs[0], task_outputs[1])
            pCVR = task_outputs[0]

            return pCTCVR, pCVR
        elif len(self.num_tasks) == 1:
            return task_outputs
        else:
            Exception("目标数目为：1或２!")


if __name__ == "__main__":
    import numpy as np

    a = torch.from_numpy(np.array([[1, 2, 4, 2, 0.5, 0.1],
                                   [4, 5, 3, 8, 0.6, 0.43],
                                   [6, 3, 2, 9, 0.12, 0.32],
                                   [9, 1, 1, 1, 0.12, 0.45],
                                   [8, 3, 1, 4, 0.21, 0.67]]))

    user_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (1, 4)}
    item_cate_dict = {'item_id': (8, 1), 'item_cate': (6, 2), 'item_num': (1, 5)}
    esmm = ESMM(user_cate_dict, item_cate_dict)
    print(esmm)
    print(esmm(a))