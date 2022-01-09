# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 下午3:44
# @Author  : gavin
# @FileName: config.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_35154281
import argparse


def get_parse():
    parse = argparse.ArgumentParser(description="Define model's parse")
    parse.add_argument("--model", default="SharedBottom", type=str, choices=["SharedBottom", "ESMM", "MMoE", "PLE"], help="MTL模型选择")
    parse.add_argument("--use_gpu", default=True, type=bool, choices=[True, False], help="是否开启GPU训练")
    parse.add_argument("--batch_size", default=128, type=int, help="批量个数")
    parse.add_argument("--lr", default=1e-3, type=float, help="学习率")
    parse.add_argument("--weight_decay", default=0, type=float, help="权重衰减系数")
    parse.add_argument("--num_works", default=4, type=int, help="线程个数")
    parse.add_argument("--seed", default=0, type=bool, help="设置随机数")
    parse.add_argument("--use_benchmark", default=False, choices=[False, True],type=bool)
    parse.add_argument("--epochs", default=100, type=int, help="迭代次数")
    parse.add_argument("--log_dir", default="./log", type=str, help="日志数据位置")
    parse.add_argument("--resume", default=False, type=bool, help="断点续传")
    parse.add_argument("--checkpoint", default=50, type=int, help="保存间隔")

    return parse.parse_args()


if __name__ == "__main__":
    args = get_parse()
    print(args.batch_size)

