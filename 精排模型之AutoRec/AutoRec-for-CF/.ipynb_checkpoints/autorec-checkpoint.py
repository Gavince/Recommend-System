import torch
import torch.nn as nn
import numpy as np
from torch.autograd import variable
from data import get_data
import math
import time
import argparse
import torch.utils.data as Data
import torch.optim as optim

class Autorec(nn.Module):
    def __init__(self,args, num_users,num_items):
        super(Autorec, self).__init__()

        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = args.hidden_units
        self.lambda_value = args.lambda_value

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items),
        )


    def forward(self,torch_input):

        encoder = self.encoder(torch_input)
        decoder = self.decoder(encoder)

        return decoder

    def loss(self,decoder,input,optimizer,mask_input):
        cost = 0
        temp2 = 0

        cost += (( decoder - input) * mask_input).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            for j in i['params']:
                #print(type(j.data), j.shape,j.data.dim())
                if j.data.dim() == 2:
                    temp2 += torch.t(j.data).pow(2).sum()

        cost += temp2 * self.lambda_value * 0.5
        return cost,rmse

def train(epoch):

    RMSE = 0
    cost_all = 0
    for step, (batch_x, batch_mask_x, batch_y) in enumerate(loader):

        batch_x = batch_x.type(torch.FloatTensor)
        batch_mask_x = batch_mask_x.type(torch.FloatTensor)

        decoder = rec(batch_x)
        loss, rmse = rec.loss(decoder=decoder, input=batch_x, optimizer=optimer, mask_input=batch_mask_x)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        cost_all += loss
        RMSE += rmse

    RMSE = np.sqrt(RMSE.detach().cpu().numpy() / (train_mask_r == 1).sum())
    print('epoch ', epoch,  ' train RMSE : ', RMSE)

def test(epoch):

    test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
    
    test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor)
    

    decoder = rec(test_r_tensor)
    #decoder = torch.from_numpy(np.clip(decoder.detach().cpu().numpy(),a_min=1,a_max=5)).cuda()

    unseen_user_test_list = list(user_test_set - user_train_set)
    unseen_item_test_list = list(item_test_set - item_train_set)

    for user in unseen_user_test_list:
        for item in unseen_item_test_list:
            if test_mask_r[user,item] == 1:
                decoder[user,item] = 3

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    print('epoch ', epoch, ' test RMSE : ', RMSE)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='I-AutoRec ')
    parser.add_argument('--hidden_units', type=int, default=500)
    parser.add_argument('--lambda_value', type=float, default=1)

    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=1)

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    data_name = 'ml-1m'
    num_users = 6040
    num_items = 3952
    num_total_ratings = 1000209
    train_ratio = 0.9

    path = "./%s" % data_name + "/"

    train_r,train_mask_r,test_r,test_mask_r,user_train_set,item_train_set,user_test_set,\
    item_test_set = get_data(path, num_users, num_items, num_total_ratings, train_ratio)

    args.cuda = torch.cuda.is_available()

    rec = Autorec(args,num_users,num_items)
    if args.cuda:
        rec.cuda()

    optimer = optim.Adam(rec.parameters(), lr = args.base_lr, weight_decay=1e-4)

    num_batch = int(math.ceil(num_users / args.batch_size))

    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r),torch.from_numpy(train_mask_r),torch.from_numpy(train_r))
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    for epoch in range(args.train_epoch):

        train(epoch=epoch)
        test(epoch=epoch)

