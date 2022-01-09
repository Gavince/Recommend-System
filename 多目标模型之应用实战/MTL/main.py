import os
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import TrainDateSet
import models
import config


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loss_fn(loss_name):
    try:
        if "mse" == loss_name.lower():
            # 针对回归
            return nn.MSELoss()
        elif "bce" == loss_name.lower():
            # 针对分类
            return nn.BCEWithLogitsLoss()
    except:
        print("损失函数不存在！！！")


def main():
    # 初始化
    args = config.get_parse()
    init_seeds(args.seed)
    start_epoch = 0

    # 加载数据
    print("Preparing data......")
    user_feat_dict = np.load('./data/Income/user_feat_dict.npy', allow_pickle=True).item()
    item_feat_dict = np.load("./data/Income/item_feat_dict.npy", allow_pickle=True).item()
    train_data = pd.read_csv("./data/Income/train_data.csv")
    val_data = pd.read_csv("./data/Income/test_data.csv")

    # 设备信息
    print("Pytorch Version: ", torch.__version__)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU: ', torch.cuda.get_device_name(0))
        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True
            print('Using cudnn.benchmark.')
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    # 设置模型参数
    model = getattr(models, args.model)(user_feat_dict, item_feat_dict)
    model.to(device)
    # 加载checkopint
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{50}_ckpt.pth')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    loss_fun1 = loss_fn("bce")
    loss_fun2 = loss_fn("bce")

    # 处理数据
    train_datasets = TrainDateSet(train_data)
    val_datasets = TrainDateSet(val_data)
    train_data_loader = DataLoader(dataset=train_datasets, batch_size=args.batch_size, num_workers=args.num_works)
    val_data_loader = DataLoader(dataset=val_datasets, batch_size=args.batch_size, num_workers=args.num_works)
    writer = SummaryWriter(args.log_dir, comment="mertics")

    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        train(model, device, train_data_loader, writer, loss_fun1, loss_fun2, epoch, args, optimizer)
        val(model, device, val_data_loader, writer, loss_fun1, loss_fun2, epoch, args)
        scheduler.step()
    writer.close()


def train(model, device, train_data_loader, writer, loss_fun1, loss_fun2, epoch, args, optimizer):
    model.train()
    y_train_income_true = []
    y_train_income_predict = []
    y_train_marry_true = []
    y_train_marry_predict = []
    total_tain_loss, count_train = 0, 0
    for x, y1, y2 in train_data_loader:
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
    print("\nEpoch %d train loss is %.3f, y1_auc is %.3f and y2_auc is %.3f" % (epoch + 1, train_loss_value,
                                                                                y1_auc, y2_auc))
    writer.add_scalar("Train loss", train_loss_value, global_step=epoch + 1)
    writer.add_scalar("Train_y1_auc", y1_auc, global_step=epoch + 1)
    writer.add_scalar("Train_y2_auc", y2_auc, global_step=epoch + 1)


def val(model, device, val_data_loader, writer, loss_fun1, loss_fun2, epoch, args):
    total_val_loss = 0
    model.eval()
    count_eval = 0

    y_val_income_true = []
    y_val_marry_true = []
    y_val_income_predict = []
    y_val_marry_predict = []

    for x, y1, y2 in val_data_loader:
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

    if epoch % args.checkpoint == 0:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.pth'.format(epoch))


if __name__ == "__main__":
    main()
