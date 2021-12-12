```python
from torch import nn
import torch
import torch.optim as optim
import numpy as np
from sklearn import datasets
```

# FM

## 原理

**为什么需要FM?**  

1、特征组合是许多机器学习建模过程中遇到的问题，如果对特征直接建模，很有可能会忽略掉特征与特征之间的关联信息，因此，可以通过构建新的交叉特征这一特征组合方式提高模型的效果。  
2、高维的稀疏矩阵是实际工程中常见的问题，并直接会导致计算量过大，特征权值更新缓慢。试想一个10000\*100的表，每一列都有8种元素，经过one-hot独热编码之后，会产生一个10000\*800的表。因此表中每行元素只有100个值为1，700个值为0。
如图：　　

<img src=./imgs/LR.png height="300" width="500" />


   而FM的优势就在于对这两方面问题的处理。首先是特征组合，通过对两两特征组合，引入交叉项特征，提高模型得分；其次是高维灾难，通过引入隐向量（对参数矩阵进行矩阵分解），完成对特征的参数估计.如图：

<img src=./imgs/FM1.png height="300" width="500" /> <img src=./imgs/FM2.png height="400" width="500" />

# FM 代码解析

## Python代码
公式推导：  
[FM算法解析及Python实现](https://www.cnblogs.com/wkang/p/9588360.html)


```python
# 初始化参数
w = zeros((n, 1))  # 其中n是特征的个数
w_0 = 0.
v = normalvariate(0, 0.2) * ones((n, k))  # 隐向量　　　　　
for it in range(self.iter): # 迭代次数
    # 对每一个样本，优化
    for x in range(m):
        # 这边注意一个数学知识：对应点积的地方通常会有sum，对应位置积的地方通常都没有，详细参见矩阵运算规则，本处计算逻辑在：http://blog.csdn.net/google19890102/article/details/45532745
        # xi·vi,xi与vi的矩阵点积
        inter_1 = dataMatrix[x] * v
        # xi与xi的对应位置乘积   与   xi^2与vi^2对应位置的乘积    的点积
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项,xi*vi*xi*vi - xi^2*vi^2
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        # 计算预测的输出
        p = w_0 + dataMatrix[x] * w + interaction
        print('classLabels[x]:',classLabels[x])
        print('预测的输出p:', p)
        # 计算sigmoid(y*pred_y)-1准确的说不是loss，原作者这边理解的有问题，只是作为更新w的中间参数，这边算出来的是越大越好，而下面却用了梯度下降而不是梯度上升的算法在
        loss = self.sigmoid(classLabels[x] * p[0, 0]) - 1
        if loss >= -1:
            loss_res = '正方向 '
        else:
            loss_res = '反方向'

        # 更新参数
        w_0 = w_0 - self.alpha * loss * classLabels[x]
        for i in range(n):
            if dataMatrix[x, i] != 0:
                w[i, 0] = w[i, 0] - self.alpha * loss * classLabels[x] * dataMatrix[x, i]
                # 计算交叉项，从而更新隐向量
                for j in range(k):
                    v[i, j] = v[i, j] - self.alpha * loss * classLabels[x] * (
                            dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
```

## Pytorch代码


```python
class FM_model(nn.Module):
    
    def __init__(self, n, k):
        
        super(FM_model, self).__init__()
        self.n = n # len(items) + len(users) n为特征数目
        self.k = k
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))

    def fm_layer(self, x):
        
      	# x 属于 R^{batch*n}
        linear_part = self.linear(x)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k) 
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2) 
        # 这里torch求和一定要用sum
        return output  # out_size = (batch, 1)

    def forward(self, x):
        
        output = self.fm_layer(x)
        
        return output
```


```python
# 输维度信息，则有
net = FM_model(100, 50)
net(torch.rand(1, 100))
```


    tensor([[-46.4297]], grad_fn=<AddBackward0>)


```python
for name, param in net.named_parameters():
    print("更新参数:" + name + ": ", param.shape)
```

    更新参数:v:  torch.Size([50, 100])
    更新参数:linear.weight:  torch.Size([1, 100])
    更新参数:linear.bias:  torch.Size([1])


## 数据


```python
num_inputs = 100
num_examples = 1
true_w = [2, -3.4, 4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples,num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

import torch.utils.data as Data
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小小批量量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```

## 训练


```python
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.3)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零,等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print(output)
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

    tensor([[-532.4891]], grad_fn=<AddBackward0>)
    epoch 1, loss: 287459.000000
    tensor([[1.9496e+12]], grad_fn=<AddBackward0>)
    epoch 2, loss: 3800834018394383212085248.000000
    tensor([[inf]], grad_fn=<AddBackward0>)
    epoch 3, loss: inf


## 参考
[点击率预估算法：FM与FFM](https://blog.csdn.net/jediael_lu/article/details/77772565?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.add_param_isCf)  
[FM算法解析及Python实现](https://www.cnblogs.com/wkang/p/9588360.html)  
[推荐系统召回四模型之：全能的FM模型](https://zhuanlan.zhihu.com/p/58160982)  
[FM：推荐算法中的瑞士军刀](https://zhuanlan.zhihu.com/p/343174108)  
[(美团)深入FFM原理与实践](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)
