import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from models import  PLE
# 定义超参数
learning_rate = 0.01
epochs = 100
count = 0
writer = SummaryWriter("../log", comment="mertics")
device = torch.device("cuda")
user_cate_dict = {'user_id': (11, 0), 'user_list': (12, 3), 'user_num': (1, 4)}
item_cate_dict = {'item_id': (8, 1), 'item_cate': (6, 2), 'item_num': (1, 5)}
model = PLE(user_feat_dict, item_feat_dict)
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fun = nn.BCEWithLogitsLoss()

train_dataload = DataLoader(train_datasets, batch_size=128, shuffle=True)
test_dataload = DataLoader(test_datasets, batch_size=128)
for epoch in tqdm(range(epochs)):
    y_train_income_true = []
    y_train_income_predict = []
    y_train_marry_true = []
    y_train_marry_predict = []
    total_loss, count = 0, 0
    for x, y1, y2 in train_dataload:
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        predict = model(x)
        y_train_income_true += list(y1.squeeze().cpu().numpy())
        y_train_marry_true += list(y2.squeeze().cpu().numpy())

        y_train_income_predict += list(
            predict[0].squeeze().cpu().detach().numpy())
        y_train_marry_predict += list(
            predict[1].squeeze().cpu().detach().numpy())

        loss1 = loss_fun(predict[0], y1.unsqueeze(1).float())
        loss2 = loss_fun(predict[1], y2.unsqueeze(1).float())
        loss = loss1 + loss2
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        count += 1

    y1_auc = roc_auc_score(y_train_income_true, y_train_income_predict)
    y2_auc = roc_auc_score(y_train_marry_true, y_train_marry_predict)
    loss_value = total_loss / count
    print("Epoch %d train loss is %.3f, y1_auc is %.3f and y2_auc is %.3f" % (epoch + 1, loss_value,
                                                                              y1_auc, y2_auc))
    writer.add_scalar("Train loss", loss_value, global_step=epoch + 1)
    writer.add_scalar("Train_y1_auc", y1_auc, global_step=epoch + 1)
    writer.add_scalar("Train_y2_auc", y2_auc, global_step=epoch + 1)

    # 验证
    total_eval_loss = 0
    model.eval()
    count_eval = 0
    y_val_income_true = []
    y_val_marry_true = []
    y_val_income_predict = []
    y_val_marry_predict = []
    for x, y1, y2 in test_dataload:
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        predict = model(x)
        y_val_income_true += list(y1.squeeze().cpu().numpy())
        y_val_marry_true += list(y2.squeeze().cpu().numpy())

        y_val_income_predict += list(
            predict[0].squeeze().cpu().detach().numpy())
        y_val_marry_predict += list(
            predict[1].squeeze().cpu().detach().numpy())
        loss_1 = loss_fun(predict[0], y1.unsqueeze(1).float())
        loss_2 = loss_fun(predict[1], y2.unsqueeze(1).float())
        loss = loss_1 + loss_2
        total_eval_loss += float(loss)
        count_eval += 1

    y1_val_auc = roc_auc_score(y_val_income_true, y_val_income_predict)
    y2_val_auc = roc_auc_score(y_val_marry_true, y_val_marry_predict)
    val_loss_value = total_eval_loss / count_eval
    print("Epoch %d val loss is %.3f, y1_auc is %.3f and y2_auc is %.3f" % (epoch + 1, val_loss_value,
                                                                            y1_auc, y2_auc))
    writer.add_scalar("Val loss", val_loss_value, global_step=epoch + 1)
    writer.add_scalar("Val_y1_auc", y1_val_auc, global_step=epoch + 1)
    writer.add_scalar("Val_y2_auc", y2_val_auc, global_step=epoch + 1)

writer.close()