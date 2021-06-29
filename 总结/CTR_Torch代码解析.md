```python
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
```

### 0.加载数据


```python
data = pd.read_csv('./criteo_sample.txt')
data["C1"]
```


    0      05db9164
    1      68fd1e64
    2      05db9164
    3      05db9164
    4      05db9164
             ...   
    195    05db9164
    196    be589b51
    197    05db9164
    198    05db9164
    199    be589b51
    Name: C1, Length: 200, dtype: object


```python
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']
```

### 1. Label Encoding for sparse features,and do simple Transformation for dense features


```python
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
```

### 2. count #unique features for each sparse field,and record dense feature field name


```python
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns)
```


```python
## 特征名称
dnn_feature_columns[1].name
```


    'C2'

### 3. generate input data for model


```python
train, test = train_test_split(data, test_size=0.2)
```


```python
# 输入类型为字典类型:{特征１：值，特征２：值}
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}
```


```python
x = [train_model_input[feature] for feature in feature_names]
x[2]
```


    85      77
    151    160
    197    153
    36     130
    127    107
          ... 
    35      61
    40      34
    64      50
    154     82
    165    146
    Name: C3, Length: 160, dtype: int64

### 4.  Define Model,train,predict and evaluate

#### 4.1 训练


```python
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

# model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
#                task='binary',
#                l2_reg_embedding=1e-5, device=device)
model = WDL(linear_feature_columns, dnn_feature_columns)
model.compile("adagrad", "binary_crossentropy",
              metrics=["binary_crossentropy", "auc"], )
model.fit(train_model_input,train[target].values,batch_size=32,epochs=10,verbose=2,validation_split=0.0)

pred_ans = model.predict(test_model_input, 256)
print("")
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
```

    cuda ready...
    cpu
    Train on 160 samples, validate on 0 samples, 5 steps per epoch
    Epoch 1/10
    0s - loss:  0.5990 - binary_crossentropy:  0.5990 - auc:  0.6504
    Epoch 2/10
    0s - loss:  0.4579 - binary_crossentropy:  0.4579 - auc:  0.9717
    Epoch 3/10
    0s - loss:  0.2952 - binary_crossentropy:  0.2952 - auc:  0.9985
    Epoch 4/10
    0s - loss:  0.1802 - binary_crossentropy:  0.1802 - auc:  0.9990
    Epoch 5/10
    0s - loss:  0.1349 - binary_crossentropy:  0.1349 - auc:  1.0000
    Epoch 6/10
    0s - loss:  0.1112 - binary_crossentropy:  0.1112 - auc:  1.0000
    Epoch 7/10
    0s - loss:  0.0968 - binary_crossentropy:  0.0968 - auc:  1.0000
    Epoch 8/10
    0s - loss:  0.0839 - binary_crossentropy:  0.0839 - auc:  1.0000
    Epoch 9/10
    0s - loss:  0.0738 - binary_crossentropy:  0.0738 - auc:  1.0000
    Epoch 10/10
    0s - loss:  0.0663 - binary_crossentropy:  0.0663 - auc:  1.0000
    
    test LogLoss 1.0295
    test AUC 0.4265

#### 4.2 WDL(Wide&Deep)模型

```python
model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
            task='binary',
            l2_reg_embedding=1e-5, device=device)
model
```

```python
WDL(
  (embedding_dict): ModuleDict(
    (C1): Embedding(27, 4)
    (C10): Embedding(142, 4)
    (C11): Embedding(173, 4)
    (C12): Embedding(170, 4)
    (C13): Embedding(166, 4)
    (C14): Embedding(14, 4)
    (C15): Embedding(170, 4)
    (C16): Embedding(168, 4)
    (C17): Embedding(9, 4)
    (C18): Embedding(127, 4)
    (C19): Embedding(44, 4)
    (C2): Embedding(92, 4)
    (C20): Embedding(4, 4)
    (C21): Embedding(169, 4)
    (C22): Embedding(6, 4)
    (C23): Embedding(10, 4)
    (C24): Embedding(125, 4)
    (C25): Embedding(20, 4)
    (C26): Embedding(90, 4)
    (C3): Embedding(172, 4)
    (C4): Embedding(157, 4)
    (C5): Embedding(12, 4)
    (C6): Embedding(7, 4)
    (C7): Embedding(183, 4)
    (C8): Embedding(19, 4)
    (C9): Embedding(2, 4)
  )
  (linear_model): Linear(
    (embedding_dict): ModuleDict(
      (C1): Embedding(27, 1)
      (C10): Embedding(142, 1)
      (C11): Embedding(173, 1)
      (C12): Embedding(170, 1)
      (C13): Embedding(166, 1)
      (C14): Embedding(14, 1)
      (C15): Embedding(170, 1)
      (C16): Embedding(168, 1)
      (C17): Embedding(9, 1)
      (C18): Embedding(127, 1)
      (C19): Embedding(44, 1)
      (C2): Embedding(92, 1)
      (C20): Embedding(4, 1)
      (C21): Embedding(169, 1)
      (C22): Embedding(6, 1)
      (C23): Embedding(10, 1)
      (C24): Embedding(125, 1)
      (C25): Embedding(20, 1)
      (C26): Embedding(90, 1)
      (C3): Embedding(172, 1)
      (C4): Embedding(157, 1)
      (C5): Embedding(12, 1)
      (C6): Embedding(7, 1)
      (C7): Embedding(183, 1)
      (C8): Embedding(19, 1)
      (C9): Embedding(2, 1)
    )
  )
  (out): PredictionLayer()
  (dnn): DNN(
    (dropout): Dropout(p=0, inplace=False)
    (linears): ModuleList(
      (0): Linear(in_features=117, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=128, bias=True)
    )
    (activation_layers): ModuleList(
      (0): ReLU(inplace=True)
      (1): ReLU(inplace=True)
    )
  )
  (dnn_linear): Linear(in_features=128, out_features=1, bias=False)
)
```

#### 4.3 compile接口

**接口调用**

```python
model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
```

**函数实现**

```python
def compile(self, optimizer, loss=None, metrics=None,):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]  # 存储所选指标名称，eg:["loss", "auc", "logloss"]
        self.optim = self._get_optim(optimizer)  # 获得指定的优化器
        self.loss_func = self._get_loss_func(loss)  # 指定损失函数
        self.metrics = self._get_metrics(metrics) # 指定评价标准（一个列表）

def _get_optim(self, optimizer):
    """ 优化器"""
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

def _get_loss_func(self, loss):
    """损失函数"""
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
                elif loss == "mse":
                    loss_func = F.mse_loss
                    elif loss == "mae":
                        loss_func = F.l1_loss
                        else:
                            raise NotImplementedError
                            else:
                                loss_func = loss
                                return loss_func
                            
def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_
```

#### 4.4 fit接口 

```python
def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
        validation_data=None, shuffle=True, callbacks=None):
    """

    :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
        dictionary mapping input names to Numpy arrays.
    :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
    :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
    :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
    :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
    :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
    :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
    :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
    :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

    """
  """
  输入前的数据类型
x = {"f1":(索引+value)(Serise类型),   "f2": data2}
  
{'C1': 85     11
 151    25
 197     0
 36     11
 127    11
        ..
 35      0
 40     24
 64     16
 154     0
 165     0
 Name: C1, Length: 160, dtype: int64, 'C2': 85      3
 151    50
 197     5
 36     13
 127    13
        ..
 35     30
 40     25
 64     32
 154     6
 165    59
 Name: C2, Length: 160, dtype: int64, 'C3': 85      77
    """
    if isinstance(x, dict):
        x = [x[feature] for feature in self.feature_index]  # 变换为[(fea1)Serise,(fea2) Serise......]
    
    # 验证集处理
    # 本身带验证集
    do_validation = False
    if validation_data:
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
        else:
            raise ValueError(
                'When passing a `validation_data` argument, '
                'it must contain either 2 items (x_val, y_val), '
                'or 3 items (x_val, y_val, val_sample_weights), '
                'or alternatively it could be a dataset or a '
                'dataset or a dataset iterator. '
                'However we received `validation_data=%s`' % validation_data)
        if isinstance(val_x, dict):
            val_x = [val_x[feature] for feature in self.feature_index]
  # 从训练数据中选择一部分数据作为验证集
    elif validation_split and 0. < validation_split < 1.:
        do_validation = True
        # x[0]为表示单个特征下数据值，存储方式为一个Serise
        if hasattr(x[0], 'shape'):
            split_at = int(x[0].shape[0] * (1. - validation_split))
        else:
            split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_arrays(x, 0, split_at),
                    slice_arrays(x, split_at))
        y, val_y = (slice_arrays(y, 0, split_at),
                    slice_arrays(y, split_at))

    else:
        val_x = []
        val_y = []
        
    # 数据处理
    # x (每一个特征（一个Serise），样本数目)
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)  # (160, )
    
    
    train_tensor_data = Data.TensorDataset(
        torch.from_numpy(
            np.concatenate(x, axis=-1)),   # concat之后数据维度变为：(样本数目，特征)  eg(160, 39)
        torch.from_numpy(y))
    if batch_size is None:
        batch_size = 256
    train_loader = DataLoader(
        dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

    print(self.device, end="\n")
    model = self.train()  # 设置为training模型
    loss_func = self.loss_func
    optim = self.optim

    sample_num = len(train_tensor_data)
    steps_per_epoch = (sample_num - 1) // batch_size + 1

    callbacks = CallbackList(callbacks)
    callbacks.set_model(self)
    callbacks.on_train_begin()
    self.stop_training = False  # used for early stopping

    # Train
    print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
        len(train_tensor_data), len(val_y), steps_per_epoch))
    for epoch in range(initial_epoch, epochs):
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        start_time = time.time()
        loss_epoch = 0
        total_loss_epoch = 0
        train_result = {}
        # 训练
        try:
            with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                for index, (x_train, y_train) in t:
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()

                    y_pred = model(x).squeeze()

                    optim.zero_grad()
                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                    reg_loss = self.get_regularization_loss()

                    total_loss = loss + reg_loss + self.aux_loss  #aux_loss为附加损失

                    loss_epoch += loss.item()
                    total_loss_epoch += total_loss.item()
                    total_loss.backward(retain_graph=True)
                    optim.step()

                    # 评估
                    if verbose > 0:
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))


        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        # Add epoch_logs
        epoch_logs["loss"] = total_loss_epoch / sample_num
        for name, result in train_result.items():
            epoch_logs[name] = np.sum(result) / steps_per_epoch

        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                epoch_logs["val_" + name] = result
        # verbose
        if verbose > 0:
            epoch_time = int(time.time() - start_time)
            print('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s - loss: {1: .4f}".format(
                epoch_time, epoch_logs["loss"])

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            print(eval_str)
        callbacks.on_epoch_end(epoch, epoch_logs)
        if self.stop_training:
            break

    callbacks.on_train_end()

```

