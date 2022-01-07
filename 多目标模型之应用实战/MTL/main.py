import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import models
import config
from data import TrainDateSet
import numpy as np
import pandas as pd

def init_seeds(seed=0):
    """"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loss_fn(loss_name):
    if "mse" == loss_name.lower():
        # 针对回归
        return nn.MSELoss()
    elif "bce" == loss_name.lower():
        # 针对分类
        return nn.BCEWithLogitsLoss()
    else:
        Exception("请输入正确的损失函数！！！")


def train():
    # 加载数据
    print("正在加载数据中......")
    user_feat_dict = np.load('./data/Income/user_feat_dict.npy', allow_pickle=True).item()
    item_feat_dict = np.load("./data/Income/item_feat_dict.npy", allow_pickle=True).item()
    train_data = pd.read_csv("./data/Income/train_data.csv")
    val_data = pd.read_csv("./data/Income/test_data.csv")
    print("加载数据已完成!")

    # 设置参数
    args = config.get_parse()
    model = getattr(models, args.model)(user_feat_dict, item_feat_dict)

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU: ', torch.cuda.get_device_name(0))
        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True
            print('Using cudnn.benchmark.')
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fun1 = loss_fn("bce")
    loss_fun2 = loss_fn("bce")
    writer = SummaryWriter("./log", comment="mertics")

    # 处理数据
    train_datasets = TrainDateSet(train_data)
    val_datasets = TrainDateSet(val_data)
    train_dataload = DataLoader(train_datasets, batch_size=args.batch_size, num_workers=args.num_works)
    val_dataload = DataLoader(val_datasets, batch_size=args.batch_size, num_workers=args.num_works)

    # 迭代训练
    for epoch in tqdm(range(args.epochs)):
        y_train_income_true = []
        y_train_income_predict = []
        y_train_marry_true = []
        y_train_marry_predict = []
        total_tain_loss, count_train = 0, 0
        for x, y1, y2 in train_dataload:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_train_income_true += list(y1.squeeze().cpu().numpy())
            y_train_marry_true += list(y2.squeeze().cpu().numpy())
            y_train_income_predict += list(
                predict[0].squeeze().cpu().detach().numpy())
            y_train_marry_predict += list(
                predict[1].squeeze().cpu().detach().numpy())

            loss1 = loss_fun1(predict[0], y1.unsqueeze(1).float())
            loss2 = loss_fun2(predict[1], y2.unsqueeze(1).float())
            loss = loss1 + loss2
            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_tain_loss += float(loss)
            count_train += 1

        y1_auc = roc_auc_score(y_train_income_true, y_train_income_predict)
        y2_auc = roc_auc_score(y_train_marry_true, y_train_marry_predict)
        train_loss_value = total_tain_loss / count_train
        print("Epoch %d train loss is %.3f, y1_auc is %.3f and y2_auc is %.3f" % (epoch + 1, train_loss_value,
                                                                                  y1_auc, y2_auc))
        writer.add_scalar("Train loss", train_loss_value, global_step=epoch + 1)
        writer.add_scalar("Train_y1_auc", y1_auc, global_step=epoch + 1)
        writer.add_scalar("Train_y2_auc", y2_auc, global_step=epoch + 1)

        # 模型验证
        total_val_loss = 0
        model.eval()
        count_eval = 0
        y_val_income_true = []
        y_val_marry_true = []
        y_val_income_predict = []
        y_val_marry_predict = []
        for x, y1, y2 in val_dataload:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_val_income_true += list(y1.squeeze().cpu().numpy())
            y_val_marry_true += list(y2.squeeze().cpu().numpy())

            y_val_income_predict += list(
                predict[0].squeeze().cpu().detach().numpy())
            y_val_marry_predict += list(
                predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_fun1(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_fun2(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_val_loss += float(loss)
            count_eval += 1

        y1_val_auc = roc_auc_score(y_val_income_true, y_val_income_predict)
        y2_val_auc = roc_auc_score(y_val_marry_true, y_val_marry_predict)
        eval_loss_value = total_val_loss / count_eval
        print("Epoch %d val loss is %.3f, y1_auc is %.3f and y2_auc is %.3f" % (epoch + 1, eval_loss_value,
                                                                                y1_val_auc, y2_val_auc))
        writer.add_scalar("Val loss", eval_loss_value, global_step=epoch + 1)
        writer.add_scalar("Val_y1_auc", y1_val_auc, global_step=epoch + 1)
        writer.add_scalar("Val_y2_auc", y2_val_auc, global_step=epoch + 1)

    writer.close()


# @torch.no_grad
# def val():
#     pass


if __name__ == "__main__":
    init_seeds(0)
    train()
