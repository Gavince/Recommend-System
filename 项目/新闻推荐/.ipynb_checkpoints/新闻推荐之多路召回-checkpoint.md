# 多路召回前言
**推荐系统流程：**  
<img src="./imgs/17393f3dbdbe7a90.png" style="zoom:50%" />   
　　所谓的“多路召回策略”就是<font color=red>指采用不同的策略、特征或者简单模型，分别召回一部分候选集，然后再把这些候选集混合在一起后供后续排序模型使用的策略</font>。然后我们来说说为啥需要用到多路召回策略，我们在设计召回层的时候，“计算速度”与“召回率”这两个指标是相互矛盾的，也就是说在提高计算速度的时候需要尽量简化召回策略，这就会导致召回率不尽人意，同样的，需要提高召回率时就需要复杂的召回策略，这样计算速度肯定会相应的降低。在权衡两者后，目前工业界普遍采用多个简单的召回策略叠加的“多路召回策略”。  
**多路召回策略：**  
<img src="./imgs/17393f42d7e616a3.png" style="zoom:45%"/>

　　在多路召回中，每个策略之间毫不相关，所以一般可以写并发多线程同时进行。例如：新闻类的推荐系统中，我们可以按文章类别、作者、热度等分别进行召回，这样召回出来的结果更贴切实际要求，同时我们可以开辟多个线程分别进行这些召回策略，这样可以更加高效。  


# 数据预处理
在一般的rs比赛中读取数据部分主要分为三种模式， 不同的模式对应的不同的数据集：
1. debug模式： 这个的目的是帮助我们基于数据先搭建一个简易的baseline并跑通， 保证写的baseline代码没有什么问题。 由于推荐比赛的数据往往非常巨大， 如果一上来直接采用全部的数据进行分析，搭建baseline框架， 往往会带来时间和设备上的损耗， 所以这时候我们往往需要从海量数据的训练集中随机抽取一部分样本来进行调试(train_click_log_sample)， 先跑通一个baseline。  
2. 线下验证模式： 这个的目的是帮助我们在线下基于已有的训练集数据， 来选择好合适的模型和一些超参数。 所以我们这一块只需要加载整个训练集(train_click_log)， 然后把整个训练集再分成训练集和验证集。 训练集是模型的训练数据， 验证集部分帮助我们调整模型的参数和其他的一些超参数。  
3. 线上模式： 我们用debug模式搭建起一个推荐系统比赛的baseline， 用线下验证模式选择好了模型和一些超参数， 这一部分就是真正的对于给定的测试集进行预测， 提交到线上， 所以这一块使用的训练数据集是全量的数据集(train_click_log+test_click_log)


```python
import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# 版本兼容
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')

data_path = './data/'
save_path = './temp_results/'
# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
metric_recall = False
```

    WARNING:tensorflow:From /home/oem/anaconda3/envs/zwynn/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term


## 读取数据


```python
# debug模式： 从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    
    return all_click
```


```python
# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data/', offline=True):
    
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    
    return all_click
```


```python
# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    
    return item_info_df
```


```python
# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    
    # 取出所有的emb向量
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open("./data/" + 'item_content_emb.pkl', 'wb'))
    
    return item_emb_dict
```


```python
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
```


```python
all_click_df = get_all_click_df(offline=False)
all_click_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>click_article_id</th>
      <th>click_timestamp</th>
      <th>click_environment</th>
      <th>click_deviceGroup</th>
      <th>click_os</th>
      <th>click_country</th>
      <th>click_region</th>
      <th>click_referrer_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>199999</td>
      <td>160417</td>
      <td>1507029570190</td>
      <td>4</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>199999</td>
      <td>5408</td>
      <td>1507029571478</td>
      <td>4</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>199999</td>
      <td>50823</td>
      <td>1507029601478</td>
      <td>4</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>199998</td>
      <td>157770</td>
      <td>1507029532200</td>
      <td>4</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>25</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>199998</td>
      <td>96613</td>
      <td>1507029671831</td>
      <td>4</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>25</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对时间戳进行归一化操作便于运算
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
```


```python
item_info_df = get_item_info_df("./data/")
item_info_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>click_article_id</th>
      <th>category_id</th>
      <th>created_at_ts</th>
      <th>words_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1513144419000</td>
      <td>168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1405341936000</td>
      <td>189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>1408667706000</td>
      <td>250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>1408468313000</td>
      <td>230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>1407071171000</td>
      <td>162</td>
    </tr>
  </tbody>
</table>
</div>




```python
item_emb_dict = get_item_emb_dict("./data/")
```

    WARNING:root:
    DeepCTR version 0.8.3 detected. Your version is 0.8.2.
    Use `pip install -U deepctr` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR/releases/tag/v0.8.3


## 获得用户-文章-时间函数
计算用户相似度是需要用到


```python
def get_user_item_time(click_df):
    """
    {user:[(item, time)......]}
    """
    
    click_df = click_df.sort_values(by="click_timestamp")
    
    def make_item_time_pair(df):
        
        return list(zip(df["click_article_id"], df["click_timestamp"]))
    
    user_item_time_df = click_df.groupby("user_id")["click_article_id", 
                                                     "click_timestamp"].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0:"item_time_list"})
    # 构建用户历史
    user_tem_time_dict = dict(zip(user_item_time_df["user_id"], user_item_time_df["item_time_list"]))
    
    return user_tem_time_dict
```

## 获得文章-用户-时间函数
计算物品相似度是需要用到


```python
def get_item_user_time_dict(click_df):
    """
    {"item":[(user, time)]......}
    """
    
    def make_user_time_pair(df):
        
        return list(zip(df['user_id'], df['click_timestamp']))
    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    
    return item_user_time_dict
```

## 获取用户历史和最后一次点击


```python
# 注：[:-1] 左闭右开
def get_hist_and_get_last_click(all_click):
    """计算历史和最后一次点击的条目"""
    
    all_click = all_click.sort_values(by=["user_id", "click_timestamp"])
    click_last_df = all_click.groupby("user_id").tail(1)
    
    def hist_func(df):
        
        if len(df) == 1:
            """只有一个用户"""
            return df
        else:
            return df[:-1]  # 用户历史点记录, 不包括最后一次点击
    
    click_hist_df = all_click.groupby("user_id").apply(hist_func).reset_index(drop=True) 
    
    return click_hist_df, click_last_df    
```

## 获取文章和用户对应的属性关系


```python
def get_item_info_dict(item_info_df):
    """
　　 获取新闻id对应的属性值，如文章所属于的类别（一篇文章可以有多个类别标签），为冷启动阶段做准备
    """
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    # 创建时间归一化
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)
    
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    
    return item_type_dict, item_words_dict, item_created_time_dict
```


```python
all_click_df.groupby('user_id')['click_article_id'].agg(set).reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>click_article_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>{30760, 157507}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>{63746, 289197}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>{168401, 36162}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>{36162, 50644}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>{39894, 42567}</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>249995</th>
      <td>249995</td>
      <td>{16129, 198659, 30730, 272143, 156560, 161425,...</td>
    </tr>
    <tr>
      <th>249996</th>
      <td>249996</td>
      <td>{160974}</td>
    </tr>
    <tr>
      <th>249997</th>
      <td>249997</td>
      <td>{123909, 183665, 124337, 96755, 181686, 124667...</td>
    </tr>
    <tr>
      <th>249998</th>
      <td>249998</td>
      <td>{235105, 160974, 236207, 237524, 202557}</td>
    </tr>
    <tr>
      <th>249999</th>
      <td>249999</td>
      <td>{160417, 162338, 235870, 95972, 352983, 156843...</td>
    </tr>
  </tbody>
</table>
<p>250000 rows × 2 columns</p>
</div>




```python
def get_user_hist_item_info_dict(all_click):
    """
    获取与用户相对应的属性值，如用户的历史点击类别、历史点击文章的平均字数
    """
    
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))
    
    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))
    
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict
```

## 获取点击次数最多的topk个文章


```python
def get_item_topk_click(clicl_df, k):
    """Topk 元素"""
    
    topk_click = clicl_df["click_article_id"].value_counts().index[:k]
    
    return topk_click
```


```python
# 获取每一篇文章的具体属性信息，凑成字典，诸如类型
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)
```


```python
# 文章信息表
item_info_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>click_article_id</th>
      <th>category_id</th>
      <th>created_at_ts</th>
      <th>words_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.978432</td>
      <td>168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0.680295</td>
      <td>189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0.689493</td>
      <td>250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>0.688942</td>
      <td>230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0.685078</td>
      <td>162</td>
    </tr>
  </tbody>
</table>
</div>



# 多路召回数据处理

## 定义多路召回的字典


```python
# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {}, 
                           'cold_start_recall': {}
                          }
```


```python
trn_hist_click_df, trn_last_click_df = get_hist_and_get_last_click(all_click_df)
```

## 召回效果评估函数  
做完了召回有时候也需要对当前的召回方法或者参数进行调整以达到更好的召回效果，因为召回的结果决定了最终排序的上限，


```python
def metics_recall(user_recall_items_dict, trn_last_clicl_df, topk=5):
    """依次召回前10, 20, 30, 40, 50个文章的击中率"""
    
    # 用户最后点击的的文章
    last_click_item_dict = dict(zip(trn_last_click_df["user_id"], trn_last_click_df["click_article_id"]))
    
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        
        for user, item in user_recall_items_dict.items():
            # 召回前k个
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
                
        # 以总用户的击中率作为指标
        hit_rate = round(hit_num *1.0 / user_num, 5)
        
        print("topk:{}, hit_num:{}, hit_rate:{}, user_num:{}".format(topk, hit_num, hit_rate, user_num))
```


```python
# 知识点
i2i_sim = {}
i2i_sim.setdefault(1, {})
for x in range(1, 5):
    # 设置默认的键值
    i2i_sim[1].setdefault("2", 0)
    i2i_sim[1]["2"] += x

i2i_sim
```




    {1: {'2': 10}}



## 物品相似度的协同过滤

此时的协同过滤添加了特定的召回规则:  
(1)用户点击的时间权重  
(2)用户点击的顺序权重  
(3)文章创建的时间权重


```python
def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
    """
    
    user_item_time_dict = get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int) # 保存喜欢item的用户数目
    
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if(i == j):
                    continue
                    
                # 考虑文章的正向顺序点击和反向顺序点击（按照时间排序而成,越靠前的文章比重越大）  
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                
                # 位置信息权重，其中的参数可以调节（距离越近，权重越大）
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                
                # 点击时间权重，其中的参数可以调节（时间越近，权重越大）
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    
    return i2i_sim_
```


```python
i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)
```

    100%|██████████| 250000/250000 [07:12<00:00, 578.18it/s] 


## 用户相似度的协同过滤
<font color="red">**注意**</font>：用户数据量较大，不容易计算出相对应的用户相似度字典,后面使用YoutubeDNN得到的用户Embedding向量进行相似度的计算。



```python
def get_user_actvivate_degree_dict(all_click_df):
    """
    统计较为活跃的用户
    """
    
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()
    
    # 将统计数据归一化，便于计算
    mm = MinMaxScaler()
    all_click_df_["click_article_id"] = mm.fit_transform(all_click_df_[["click_article_id"]])
    user_activate_degree_dict = dict(zip(all_click_df_["user_id"], all_click_df_["click_article_id"]))
    
    return user_activate_degree_dict
```


```python
def usercf_sim(all_click_df, user_activate_degree_dict):
    """
    基于用户活跃度(规则)的用户相似度计算
    """
    
    item_user_time_dict = get_item_user_time_dict(all_click_df)
    
    u2u_sim = {}
    user_cnt = defaultdict(int)
    
    for item, user_item_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_item_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_item_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                    
                # 引入用户活跃度    
                activate_weight = 100*0.5*(user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_item_list)+1)
                
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wuv in related_users.items():
            u2u_sim_[u][v] = wuv/math.sqrt(user_cnt[u], user_cnt[v])
    
    pickle.dump(u2u_sim_, open("./data/" + "usercf_u2u_sim.pkl", "wb"))
    
    return u2u_sim_
```


```python
user_activate_degree_dict =  get_user_actvivate_degree_dict(all_click_df)
# u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)
```

## 物品Embedding相似
每一篇文章有一个Embedding向量，通过计算不同文章的Embedding相似度，候选出Topk篇文章。


```python
# 知识点
np.arange(1, 2)
```




    array([1])




```python
# 基于文章内容的相似度计算
def embediing_sim(click_df, item_emb_df, save_path, topk = 5):
    """
    通过Embedding计算物品相似度
    """
    
    # 索引与文章id映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df["article_id"]))
    item_emb_cols = [x for x in item_emb_df.columns if "emb" in x]
    
    # 向量l2范数归一化
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    item_emb_np = item_emb_np/ np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    
    # 使用faiss建立索引，加快相似度的计算
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    sim, idx = item_index.search(item_emb_np, topk)
    
    item_sim_dict = collections.defaultdict(dict)  # TODO:查阅资料两层字典的结构{{},......}
    
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        # 真实文章id 
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 最相似的物品是其本身,即为第一个元素, topk-1个商品被候选
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            # 得到相似的文章id
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value  # {target_raw_id:{rele_raw_id:sim_value}}
            
    
    pickle.dump(item_sim_dict, open(save_path + "emb_i2i_sim.pkl", "wb"))
    
    return item_sim_dict
```


```python
item_emb_df = pd.read_csv("./data/articles_emb.csv")
emb_i2i_sim = embediing_sim(all_click_df, item_emb_df, save_path, topk=10)
```

    364047it [00:14, 24709.00it/s]


# 多路召回


## 数据采样（重点）
**目的**：对每一个用户的历史点击文章大标签，容易造成数据缺乏多样性，因此，通过滑动窗口的方式构建负样本，保证用户数据的多样性。  
<img src="./imgs/recall.png" style="zoom:60%" /> 


```python
# 知识点
a = np.arange(1,12)
a[::-1]
```




    array([11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])




```python
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data["click_article_id"].unique()
    
    train_set = []
    test_set = []
    
    for reviewID, hist in tqdm(data.groupby("user_id")):
        pos_list = hist["click_article_id"].tolist()
        
        if negsample > 0: # 对每一正样本进行negsample负样本采样
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True)
            
        # 特例，长度为1时需要考虑
        if len(pos_list) == 1:
            train_set.append((reviewID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
            test_set.append((reviewID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
        
        # 滑动窗口构造正负样本, 按照时间顺序，将最后一次访问的文章设置标签为１的正样本
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            
            if i != len(pos_list) - 1:
                # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                train_set.append((reviewID, hist[::-1], pos_list[i], 1, len(hist[::-1])))
                # 为每一个用户构建negsample个负样本
                for negi in range(negsample):
                    # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
                    train_set.append((reviewID, hist[::-1], neg_list[i*negsample+negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewID, hist[::-1], pos_list[i], 1, len(hist[::-1])))
                    
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set
```


```python
# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}

    return train_model_input, train_label
```

## YoutubeDNN

- 模型结构

![](./imgs/image-20201111160516562.png)

- 模型源码（[代码分析链接](https://blog.csdn.net/weixin_35154281/article/details/112493756)）


```python
def YoutubeDNN_(user_feature_columns, item_feature_columns, num_sampled=5,
               user_dnn_hidden_units=(64, 32),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, output_activation='linear', seed=1024, ):
    
    """Instantiates the YoutubeDNN Model architecture.
    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param output_activation: Activation function to use in output layer
    :return: A Keras model instance.
    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now YoutubeNN only support 1 item feature like item_id")
    item_feature_name = item_feature_columns[0].name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation=output_activation, seed=seed)(user_dnn_input)

    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

    item_embedding_matrix = embedding_matrix_dict[
        item_feature_name]
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))

    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [pooling_item_embedding_weight, user_dnn_out, item_features[item_feature_name]])
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)  # 设置新的属性值
    model.__setattr__("user_embedding", user_dnn_out) # Embedding

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model
```

- 实际应用


```python
def youtubednn_u2i_dict(data, topk=20):
    """
    使用YoutubeDNN进行新闻召回计算
    """
    
    sparse_feature = ["click_article_id", "user_id"]
    SEQ_LEN = 30 # paddding
    
    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')  
    
    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {} # 存储相对应的类别数目
    
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1 # 保证Embedding向量的维度一致
    
    user_profile = data[["user_id"]].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')
    
    # 标签前和标签后的索引对比
    user_index_2_rawid = dict(zip(user_profile["user_id"], user_profile_["user_id"]))
    item_index_2_rawid = dict(zip(item_profile["click_article_id"], item_profile_["click_article_id"]))
    
    # 划分数据和整理数据集
    train_set, test_set = gen_data_set(data, negsample=0)
    
    train_model_input, train_label = gen_model_input(train_set, user_profile, seq_max_len=SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, seq_max_len=SEQ_LEN)
    
    # Embedding维度
    embedding_dim = 16
    # 用户特征
    user_feature_columns = [SparseFeat("user_id", feature_max_idx["user_id"], embedding_dim=embedding_dim),
                                   VarLenSparseFeat(SparseFeat("hist_article_id", feature_max_idx["click_article_id"]
                                                               , embedding_dim=embedding_dim, embedding_name="click_article_id"), SEQ_LEN, "mean", length_name="hist_len"),]  # 变量可变，选择设置为固定大小的序列长度，可使用均值填补
    # 物品特征
    item_feature_columns = [SparseFeat("click_article_id", feature_max_idx["click_article_id"], embedding_dim=embedding_dim)]
    
    # trianing model
    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5, user_dnn_hidden_units=(64, embedding_dim))
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    history = model.fit(train_model_input, train_label, batch_size = 256, epochs=1, verbose=1, validation_split=0.0) # 不设置验证集，全数据进行训练
    
    # 得到物品和用户向量
    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id":item_profile["click_article_id"].values}
    
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    
    user_embs = user_embedding_model.predict(test_model_input, batch_size=2*12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2**12)
    
    # L2范数归一化
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
    
    # 保存以字典的形式进行
    raw_user_id_emb_dict = {user_index_2_rawid[k]: v for k, v in zip(user_profile["user_id"], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: v for k, v in zip(item_profile["click_article_id"], item_embs)}
    pickle.dump(raw_user_id_emb_dict, open(save_path + "user_youtube_emb.pkl", "wb"))
    pickle.dump(raw_item_id_emb_dict, open(save_path + "item_youtube_emb.pkl", "wb"))
    
    # 召回
    # faiss 搜索，对每一个用户搜索召回最近的K个商品,最近邻搜索
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    # 搜索
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)
    user_recall_item_dict = collections.defaultdict(dict)
    
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input["user_id"], sim, idx)):
        target_raw_idx = user_index_2_rawid[target_idx]
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):  # 首元素为自身
            rele_raw_idx = item_index_2_rawid[rele_idx]
            #　得到用户相关物品去与其相对应的相似度数值
            user_recall_item_dict[target_raw_idx][rele_raw_idx] = user_recall_item_dict.get(target_idx, {}).get(rele_raw_idx, 0) + sim_value
            
    # 对召回结果进行排序,按照相似性从大到小进行排列{k1:{v1:smi2, v2:smi2, ......}, ......}
    user_recall_item_dict = {k:sorted(v.items(), key=lambda x:x[1], reverse=True) for k, v in user_recall_item_dict.items()}
    
    pickle.dump(user_recall_item_dict, open(save_path + "youtube_u2i_dict.pkl", "wb"))
    
    return user_recall_item_dict
```


```python
# 召回数据
metric_recall = False

if not metric_recall:
    user_multi_recall_dict["youtubednn_recall"] = youtubednn_u2i_dict(all_click_df, topk=20)
else:
    trn_hist_click, trn_last_click_df = get_hist_and_last_click(all_click_df)
    user_multi_recall_dict["youtubednn_recall"] = youtubednn_u2i_dict(trn_hist_click, topk=20)
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)
```

    100%|██████████| 250000/250000 [00:33<00:00, 7489.45it/s]


    WARNING:tensorflow:From /home/oem/anaconda3/envs/zwynn/lib/python3.8/site-packages/tensorflow/python/keras/initializers/initializers_v1.py:47: calling RandomNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor


    WARNING:tensorflow:From /home/oem/anaconda3/envs/zwynn/lib/python3.8/site-packages/tensorflow/python/keras/initializers/initializers_v1.py:47: calling RandomNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor


    Train on 1149673 samples
    1149673/1149673 [==============================] - 265s 231us/sample - loss: 0.1348
    WARNING:tensorflow:From /home/oem/anaconda3/envs/zwynn/lib/python3.8/site-packages/tensorflow/python/keras/engine/training_v1.py:2070: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.


    WARNING:tensorflow:From /home/oem/anaconda3/envs/zwynn/lib/python3.8/site-packages/tensorflow/python/keras/engine/training_v1.py:2070: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    250000it [00:21, 11627.67it/s]


## ItemCF Recall的召回操作
上面已经通过协同过滤，Embedding检索的方式得到了文章的相似度矩阵，下面使用协同过滤的思想，给用户召回与其历史文章相似的文章。 这里在召回的时候，也是用了<font color="red">**关联规则的方式**</font>：

1. 考虑相似文章与历史点击文章顺序的权重(细节看代码)
2. 考虑文章创建时间的权重，也就是考虑相似文章与历史点击文章创建时间差的权重
3. 考虑文章内容相似度权重(使用Embedding计算相似文章相似度，但是这里需要注意，在Embedding的时候并没有计算所有商品两两之间的相似度，所以相似的文章与历史点击文章不存在相似度，需要做特殊处理)


```python
def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    
    user_item_time_dict = get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if(i == j):
                    continue
                    
                # 考虑文章的正向顺序点击和反向顺序点击    
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    
    return i2i_sim_
```


```python
i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)
```

    100%|██████████| 250000/250000 [06:56<00:00, 599.91it/s] 



```python
# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵
        
        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    
    # 记录召回的文章
    item_rank = {}
    
    # 遍历用户历史喜欢文章
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
            
            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc))
            
            # Embedding相似度
            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]
                
            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij
    
    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank
```

## 评估召回


```python
# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, \
                                                        i2i_sim, sim_item_topk, recall_item_num, \
                                                        item_topk_click, item_created_time_dict, emb_i2i_sim)

user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path + 'itemcf_recall_dict.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)
```

    100%|██████████| 250000/250000 [1:13:28<00:00, 56.71it/s]  


## Embedding CF 召回的操作


```python
# 使用物品之间的Embedding相似度矩阵来计算相对应的物品召回
if metric_recall:
    trn_hist_click_df, trn_last_click_df = gen_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
# 将用户Embedding相似度
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))


sim_item_topk = 20  # 相似物品
recall_item_num = 10 # 召回物品

# 热度文章
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

# 热度文章
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df["user_id"].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
    
user_multi_recall_dict["embedding_sim_item_recall"] = user_recall_items_dict
pickle.dump(user_multi_recall_dict["embedding_sim_item_recall"], open(save_path + "embedding_sim_item_recall.pkl", "wb"))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)
```

    100%|██████████| 250000/250000 [01:44<00:00, 2383.54it/s]


## UserCF的召回操作
基于用户协同过滤，核心思想是给用户推荐与其相似的用户历史点击文章，因为这里涉及到了相似用户的历史文章，这里仍然可以加上一些关联规则来给用户可能点击的文章进行加权，这里使用的关联规则主要是考虑相似用户的历史点击文章与被推荐用户历史点击商品的关系权重，而这里的关系就可以直接借鉴基于物品的协同过滤相似的做法，只不过这里是对被推荐物品关系的一个累加的过程，下面是使用的一些关系权重，及相关的代码：

1. <font color="red">**算被推荐用户历史点击文章与相似用户历史点击文章的相似度，文章创建时间差，相对位置的总和，作为各自的权重**</font>


```python
# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num, 
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵
        
        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]    #  [(item1, time1), (item2, time2)..]
    user_hist_items = set([i for i, t in user_item_time_list])   # 存在一个用户与某篇文章的多次交互， 这里得去重
    
    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)
            
            
            # 考虑规则的召回
            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0
            
            # 当前文章　　与　　该用户看的历史文章　　进行一个权重交互（权重加和）
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重（点击位置越靠近，权重越大，其以按照时间先后顺序进行排列）
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]
                
                # 创建时间差权重（时间差越大，其对应的权重值变小）
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                
            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv
        
    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items(): # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100 # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break
        
    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]    
    
    return items_rank
```


```python
# # 此处无真实的的用户相似度字典，不进行计算
# if metric_recall:
#     trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
# else:
#     trn_hist_click_df = all_click_df
    
# user_recall_items_dict = collections.defaultdict(dict)
# user_item_time_dict = get_user_item_time(trn_hist_click_df)

# u2u_sim = pickle.load(open(save_path + 'usercf_u2u_sim.pkl', 'rb'))

# sim_user_topk = 20
# recall_item_num = 10
# item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
#                                                         recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)    

# pickle.dump(user_recall_items_dict, open(save_path + 'usercf_u2u2i_recall.pkl', 'wb'))

# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)
```

## User_embedding 召回

**目的**：通过使用YoutubeDNN计算得到的用户embedding向量来计算用户相似度矩阵，从而得到基于规则的用户CF召回。


```python
def u2u_embedding_sim(click_df, user_emb_dict , save_path, topk):
    """通过Embedding计算用户相似度"""
    
    user_list = []
    user_emb_list = []
    
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)
        
    # 存储装换索引对应关系
    user_index_2_rawid_dict = {k:v for k, v in zip(range(len(user_list)), user_list)}
    user_emb_np = np.array(user_emb_list)
    
    # faiss搜做
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    sim, idx = user_index.search(user_emb_np, topk)
    
    # 存储相似度
    user_sim_dict = collections.defaultdict(dict)
    
    # 获取用户相似度
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_idx = user_index_2_rawid_dict[target_idx]
        
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_idx = user_index_2_rawid_dict[rele_idx]
            
            user_sim_dict[target_raw_idx][rele_raw_idx] = user_sim_dict.get(target_raw_idx, {}).get(rele_raw_idx, 0) + sim_value
            
    pickle.dump(user_sim_dict, open(save_path + "youtube_u2u_sim.pkl", "wb"))
    
    return user_sim_dict
```


```python
# 存储通过YoutubeDNN存放的用户embedding向量
user_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))

# 计算相似度
u2u_sim = u2u_embedding_sim(all_click_df, user_emb_dict, save_path, topk=10)
```

    250000it [00:11, 22080.02it/s]



```python
# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(save_path + 'youtube_u2u_sim.pkl', 'rb'))

sim_user_topk = 20
recall_item_num = 10

# 热门新闻
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
    
user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'], open(save_path + 'youtubednn_usercf_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)
```

    100%|██████████| 250000/250000 [08:48<00:00, 473.43it/s] 


## 冷启动问题

**冷启动问题可以分成三类：文章冷启动，用户冷启动，系统冷启动。**

- 文章冷启动：对于一个平台系统新加入的文章，<font color="red">该文章没有任何的交互记录</font>，如何推荐给用户的问题。(对于我们场景可以认为是，日志数据中没有出现过的文章都可以认为是冷启动的文章)
- 用户冷启动：对于一个平台系统新来的用户，该用户还没有文章的交互信息，如何给该用户进行推荐。(对于我们场景就是，测试集中的用户是否在测试集对应的log数据中出现过，如果没有出现过，那么可以认为该用户是冷启动用户。但是有时候并没有这么严格，我们也可以自己设定某些指标来判别哪些用户是冷启动用户，比如通过使用时长，点击率，留存率等等)
- 系统冷启动：就是对于一个平台刚上线，还没有任何的相关历史数据，此时就是系统冷启动，其实也就是前面两种的一个综合。

**当前场景下冷启动问题的分析：**

对当前的数据进行分析会发现，日志中所有出现过的点击文章只有3w多个，而整个文章库中却有30多万，那么测试集中的用户最后一次点击是否会点击没有出现在日志中的文章呢？如果存在这种情况，说明用户点击的文章之前没有任何的交互信息，这也就是我们所说的文章冷启动。通过数据分析还可以发现，测试集用户只有一次点击的数据占得比例还不少，其实仅仅通过用户的一次点击就给用户推荐文章使用模型的方式也是比较难的，这里其实也可以考虑用户冷启动的问题，但是这里只给出物品冷启动的一些解决方案及代码，关于用户冷启动的话提一些可行性的做法。

1. 文章冷启动(没有冷启动的探索问题)    
   其实我们这里不是为了做文章的冷启动而做冷启动，而是猜测用户可能会点击一些没有在log数据中出现的文章，我们要做的就是如何从将近27万的文章中选择一些文章作为用户冷启动的文章，这里其实也可以看成是一种召回策略，我们这里就采用简单的比较好理解的基于规则的召回策略来获取用户可能点击的未出现在log数据中的文章。
   现在的问题变成了：如何给每个用户考虑从27万个商品中获取一小部分商品？随机选一些可能是一种方案。下面给出一些参考的方案。
   1. 首先基于Embedding召回一部分与用户历史相似的文章
   2. 从基于Embedding召回的文章中通过一些规则过**滤掉**一些文章，使得留下的文章用户更可能点击。<font color = "red">我们这里的规则，可以是，留下那些与用户历史点击文章主题相同的文章，或者字数相差不大的文章。并且留下的文章尽量是与测试集用户最后一次点击时间更接近的文章，或者是当天的文章也行。</font>
2. 用户冷启动    
   这里对测试集中的用户点击数据进行分析会发现，测试集中有百分之20的用户只有一次点击，那么这些点击特别少的用户的召回是不是可以单独做一些策略上的补充呢？或者是在排序后直接基于规则加上一些文章呢？这些都可以去尝试，这里没有提供具体的做法。
   
**注意：**   

这里看似和基于embedding计算的item之间相似度然后做itemcf是一致的，但是现在我们的目的不一样，我们这里的目的是找到相似的向量，并且还没有出现在log日志中的商品，再加上一些其他的冷启动的策略，这里需要找回的数量会偏多一点，不然被筛选完之后可能都没有文章了


```python
# 通过embedding相似度计算与之相对应的新闻推荐，由于embedding为总文章的向量，所以可以计算推荐未在训练样本中所出现的数据，即物品冷启动
# 不做召回评估
trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + "emb_i2i_sim.pkl", "rb"))

sim_item_topk = 150
recall_item_num = 100 # 召回数目

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df["user_id"].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk
                                                        , recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
    
pickle.dump(user_recall_items_dict, open(save_path + "cold_start_items_raw_dict.pkl", "wb"))
```

    100%|██████████| 250000/250000 [02:02<00:00, 2040.02it/s]



```python
# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果

def get_click_article_ids_set(all_click_df):
    return set(all_click_df.click_article_id.values)

def cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                     user_last_item_created_time_dict, item_type_dict, item_words_dict, 
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    """
        冷启动的情况下召回一些文章
        :param user_recall_items_dict: 基于内容embedding相似性召回来的很多文章， 字典， {user1: [(item1:score1), ..], }
        :param user_hist_item_typs_dict: 字典， 用户点击的文章的主题映射
        :param user_hist_item_words_dict: 字典， 用户点击的历史文章的字数映射
        :param user_last_item_created_time_idct: 字典，用户点击的历史文章创建时间映射
        :param item_tpye_idct: 字典，文章主题映射
        :param item_words_dict: 字典，文章字数映射
        :param item_created_time_dict: 字典， 文章创建时间映射
        :param click_article_ids_set: 集合，用户点击过得文章, 也就是日志里面出现过的文章
        :param recall_item_num: 召回文章的数量， 这个指的是没有出现在日志里面的文章数量
    """
    
    cold_start_user_items_dict = {}
    # 对用用户Embdedding召回的文章进行冷启动分析
    for user, item_list in tqdm(user_recall_items_dict.items()):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            # 获取历史文章信息
            hist_item_type_set = user_hist_item_typs_dict[user]
            hist_mean_words = user_hist_item_words_dict[user] #平均阅读字数
            hist_last_item_created_time = user_last_item_created_time_dict[user]
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)
            
            # 获取当前召回文章的信息
            curr_item_type = item_type_dict[item]
            curr_item_words = item_words_dict[item]
            curr_item_created_time = item_created_time_dict[item]
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)
            
            # 新闻不能出现在已有的日志记录中,eg:类型，点击文章，字数，创建时间
            if  curr_item_type not in hist_item_type_set or \
                item not in click_article_ids_set or \
                abs(curr_item_words - hist_mean_words) > 200 or \
                abs((curr_item_created_time-hist_last_item_created_time).days)>90:
                continue
            
            cold_start_user_items_dict[user].append((item, score))
        
    # 控制冷启动所召回的数目
    cold_start_user_items_dict = {k : sorted(v, key=lambda x:x[1], reverse=True)[:recall_item_num] for k, v in cold_start_user_items_dict.items()}
    pickle.dump(cold_start_user_items_dict, open(save_path + "cold_start_user_items_dict.pkl", "wb"))

    return cold_start_user_items_dict

```


```python
all_click_df_ = all_click_df.copy()
all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')

user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = get_user_hist_item_info_dict(all_click_df_)

click_article_ids_set = get_click_article_ids_set(all_click_df)
# 需要注意的是
# 这里使用了很多规则来筛选冷启动的文章，所以前面再召回的阶段就应该尽可能的多召回一些文章，否则很容易被删掉
cold_start_user_items_dict = cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                                              user_last_item_created_time_dict, item_type_dict, item_words_dict, \
                                              item_created_time_dict, click_article_ids_set, recall_item_num)

user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict
```

    100%|██████████| 250000/250000 [01:08<00:00, 3630.68it/s]


## 多路召回合并
多路召回合并就是将前面所有的召回策略得到的用户文章列表合并起来，下面是对前面所有召回结果的汇总
1. 基于itemcf计算的item之间的相似度sim进行的召回 
2. 基于embedding搜索得到的item之间的相似度进行的召回
3. YoutubeDNN召回
4. YoutubeDNN得到的user之间的相似度进行的召回
5. 基于冷启动策略的召回
**策略**：[说明链接](https://www.bilibili.com/video/av74918793/)  
![](./imgs/recall+.png)  
**注意：**    
在做召回评估的时候就会发现有些召回的效果不错有些召回的效果很差，所以对每一路召回的结果，我们可以认为的定义一些权重，来做最终的相似度融合


```python
def combine_recall_results(user_multi_recall_dict, weigh_dict=None, topk=25):
    """融合不同的召回策略"""
    
    final_recall_items_dict = {}
    
    def norm_user_recall_item_sim(sorted_item_dict):
        """归一化相似度，便于最后的计算"""
        
        # 召回数量为0或１
        if len(sorted_item_dict) < 2:
            return sorted_item_dict
        
        # 进行归一化
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        norm_sorted_item_list = []
        
        for item, score in sorted_item_dict:
            if max_sim > 0:
                norm_score = 1.0*(score - min_sim)/(max_sim - min_sim)
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
        
        return norm_sorted_item_list
    
    print("多路召回合并......")
    
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + "......")
        
        # 设置召回权重
        if weight_dict is None:
            recall_method_weight = 1.0
        else:
            recall_method_weight = weight_dict[method]
        # 归一化
        for user_id, sorted_item_list in user_recall_items.items():
            user_recall_items[user_id] = norm_user_recall_item_sim(sorted_item_list)
            
        # 计算召回
        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score  #　不同召回方法下相应的权重值进行累加
        
    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))

    return final_recall_items_dict_rank       
```


```python
# 这里直接对多路召回的权重给了一个相同的值，其实可以根据前面召回的情况来调整参数的值,加权平均召回
weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
               'embedding_sim_item_recall': 1.0,
               'youtubednn_recall': 1.0,
               'youtubednn_usercf_recall': 1.0, 
               'cold_start_recall': 1.0}
```


```python
# 最终合并之后每个用户召回150个商品进行排序
final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)
```

      0%|          | 0/5 [00:00<?, ?it/s]

    多路召回合并......
    itemcf_sim_itemcf_recall......


     20%|██        | 1/5 [00:06<00:26,  6.55s/it]

    embedding_sim_item_recall......


     40%|████      | 2/5 [00:12<00:18,  6.22s/it]

    youtubednn_recall......


     60%|██████    | 3/5 [00:27<00:17,  8.98s/it]

    youtubednn_usercf_recall......


     80%|████████  | 4/5 [00:33<00:08,  8.20s/it]

    cold_start_recall......


    100%|██████████| 5/5 [01:02<00:00, 12.52s/it]


# 参考
[YoutubeDNN召回理论加解释](https://mp.weixin.qq.com/s?src=11&timestamp=1608077602&ver=2769&signature=IA7HUMD9kNrVGxYC6MFdo2oBVjzhJoCiPYLF2KkXAZif7Tr1Vq5lQ3ZMAkoQac0fghXy*WOO38QvJND71nK0k6JBYg8892c4cx2tncaTV-yq-HVwbm5EF7VgnAfKW-kN&new=1)  
[(Yourubednn运行时的一个错误)SymbolicException: Inputs to eager execution function cannot be Keras symbolic tensors](https://blog.csdn.net/weixin_42295205/article/details/110870741?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-2&spm=1001.2101.3001.4242)  
[推荐系统召回策略之多路召回与Embedding召回](https://juejin.cn/post/6854573221707317261)  
[推荐系统 embedding 技术实践总结](https://zhuanlan.zhihu.com/p/143763320)  
[多路召回的融合排序](https://blog.csdn.net/RenBinSmile/article/details/104767475)  
[推荐系统怎样实现多路召回的融合排序](https://www.bilibili.com/video/av74918793/)
