## 协同过滤（基于物品和品牌）

主要对用户行为表进行处理，从而构建用户与物品表，进而通过协同过滤的方法获得相对应的推荐．

### 环境配置

``` python
import pandas as pd
import numpy as np
import pyspark
import os 
import datetime
import time
from pyspark import SparkConf
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession \
    .builder \
    .config('spark.executor.memory','16g')\
    .config('spark.driver.memory','8g')\
    .config('spark.driver.maxResultsSize','0')\
    .getOrCreate()
```

### EDA

- **数据获取（原始数据量过大，所以选择切分一小部分数据进行探索分析）**

  ```python
  def get_small_data():
      """选取100000条数据进行处理"""
      
      reads  = pd.read_csv("./data/behavior_log.csv", chunksize=100, iterator=True)
      for i, read in enumerate(reads):
          if i == 0:
              read.to_csv("./data/small_behavior.csv", index=False)
          elif i<1000:  # 保存10万行数据
              # 以追加的形式读取数据
              read.to_csv("./data/small_behavior.csv", mode="a", index=False, header=False)
          else:
              break
              
      return 0
  ```


- **读取数据并分析**

  ①读取数据

  ```python
  # 用户行为表(主表)
  behavior = spark.read.csv("./data/small_behavior.csv", header=True)
  behavior.show(5)
  ```

  ```python
  +------+----------+----+-----+------+
  |  user|time_stamp|btag| cate| brand|
  +------+----------+----+-----+------+
  |558157|1493741625|  pv| 6250| 91286|
  |558157|1493741626|  pv| 6250| 91286|
  |558157|1493741627|  pv| 6250| 91286|
  |728690|1493776998|  pv|11800| 62353|
  |332634|1493809895|  pv| 1101|365477|
  +------+----------+----+-----+------+
  only showing top 5 rows
  ```

  ```python
  # 查看数据类型与样本量
  behavior.printSchema(), behavior.count()
  ```

  ```python
  root
   |-- user: string (nullable = true)
   |-- time_stamp: string (nullable = true)
   |-- btag: string (nullable = true)
   |-- cate: string (nullable = true)
   |-- brand: string (nullable = true)
   (None, 100000)
  ```

  ②修改字段类型

  ```python
  from pyspark.sql.types import StructType, StringType, StructField, IntegerType, LongType
  
  # 修改数据类型
  schema = StructType([StructField("userId", IntegerType())
      , StructField("timestamp", LongType())
      , StructField("btag", StringType())
      , StructField("cateId", IntegerType())
      , StructField("brandId", IntegerType())
      ])
  # 重新加载数据，并赋予相对应的数据类型
  behavior = spark.read.csv("./data/small_behavior.csv", header=True, schema=schema)
  behavior.printSchema()
  ```

  ```python
  root
   |-- userId: integer (nullable = true)
   |-- timestamp: long (nullable = true)
   |-- btag: string (nullable = true)
   |-- cateId: integer (nullable = true)
   |-- brandId: integer (nullable = true)
  ```

  ③统计

  ```python
  ## 分析用户行为表中的信息
  print("缺失情况：", behavior.dropna().count()/behavior.count())
  print("用户信息人数:", behavior.groupby("userId").count().count())
  print("品牌种类：", behavior.groupby("brandId").count().count())
  print("商品种类：", behavior.groupby("cateId").count().count())
  print("行为类型:", behavior.groupby("btag").count().collect())  # 会存入内存
  behavior.groupby("btag").count().toPandas().plot(kind="bar")
  plt.xticks(range(4), ["buy", "fav", "cart", "pv"])
  plt.show()
  ```

  ```python
  缺失情况： 1.0
  用户信息人数: 36216
  品牌种类： 8141
  商品种类： 2520
  行为类型: [Row(btag='buy', count=1202), Row(btag='fav', count=1216), Row(btag='cart', count=2055), Row(btag='pv', count=95527)]
  ```

  ![](./imgs/output_13_1.png)

- 获得用户与商品和品牌表

  ①数据映射

  ```python
  #  每位用户对每一类商品的各种类型进行统计
  count_cate = behavior.groupBy(behavior.userId, behavior.cateId).pivot("btag", ["pv","fav","cart","buy"]).count()
  # 不同品牌
  count_brand = behavior.groupBy(behavior.userId, behavior.brandId).pivot("btag", ["pv","fav","cart","buy"]).count()
  count_brand.show(5),  count_cate.show(5)
  ```

  ```python
  +-------+------+---+----+----+----+
  | userId|cateId| pv| fav|cart| buy|
  +-------+------+---+----+----+----+
  |  62553|  6423|  1|null|null|null|
  |1088376|  6142|  1|null|null|null|
  | 534566|  5247|  1|null|null|null|
  | 903066|  5868|  1|null|null|null|
  | 816240|  2468|  1|null|null|null|
  +-------+------+---+----+----+----+
  only showing top 5 rows
  
  +-------+-------+---+----+----+----+
  | userId|brandId| pv| fav|cart| buy|
  +-------+-------+---+----+----+----+
  | 562184| 311057|  1|null|null|null|
  |  20612| 184921|  3|null|null|null|
  |1078294| 113336| 14|null|null|null|
  | 676744|   6019|  1|   1|null|null|
  | 446325| 153954|  1|null|null|null|
  +-------+-------+---+----+----+----+
  only showing top 5 rows
  ```

  ②保存数据与类型变换

  ```python
  # 保存用户商品评分表
  if os.path.exists("data/count_brand.csv") and os.path.exists("data/count_cate.csv/"):
      print("存储文件已存在！")
  else:
      count_cate.write.csv("data/count_cate.csv", header=True)
      count_brand.write.csv("data/count_brand.csv", header=True)
  ```

  ③cateId字段

  ```python
  #  字段类型装换
  schema = StructType([
      StructField("userId", IntegerType()),
      StructField("cateId", IntegerType()),
      StructField("pv", IntegerType()),
      StructField("fav", IntegerType()),
      StructField("cart", IntegerType()),
      StructField("buy", IntegerType())
  ])
  count_cate_df = spark.read.csv("data/count_cate.csv", header=True, schema=schema)
  count_cate_df.show(5), count_cate_df.printSchema()
  ```

  ```python
  +-------+------+---+----+----+----+
  | userId|cateId| pv| fav|cart| buy|
  +-------+------+---+----+----+----+
  | 754250|  6339|  1|null|null|null|
  |1053070|  4505|  4|null|null|null|
  | 148496|  6407|  1|null|null|null|
  | 184539|  4505|  3|   1|null|null|
  |  96267|  4267|  1|null|null|null|
  +-------+------+---+----+----+----+
  only showing top 5 rows
  
  root
   |-- userId: integer (nullable = true)
   |-- cateId: integer (nullable = true)
   |-- pv: integer (nullable = true)
   |-- fav: integer (nullable = true)
   |-- cart: integer (nullable = true)
   |-- buy: integer (nullable = true)
  ```

  ④brandId字段

  ```python
  schema = StructType([
      StructField("userId", IntegerType()),
      StructField("brandId", IntegerType()),
      StructField("pv", IntegerType()),
      StructField("fav", IntegerType()),
      StructField("cart", IntegerType()),
      StructField("buy", IntegerType())
  ])
  count_brand_df = spark.read.csv("data/count_brand.csv", header=True, schema=schema)
  count_brand_df.show(5)
  ```

  ```python
  +------+-------+---+----+----+----+
  |userId|brandId| pv| fav|cart| buy|
  +------+-------+---+----+----+----+
  |911367| 123654|  4|null|null|null|
  |356212| 271250|  2|null|null|null|
  |837910| 286075|  1|null|null|null|
  |445701|  95766|  1|null|null|null|
  |568859| 132678|  4|null|null|null|
  +------+-------+---+----+----+----+
  only showing top 5 rows
  ```

### ALS建模

- **构建ALS模型（基于商品类型）**

  ![](./imgs/LFM矩阵分解图解.png)

  ①计算rating

  ```python
  def process_row(row):
      """获取用户商品评分数据"""
      
      # 获取指定次数
      pv_count = row.pv if row.pv else 0.0
      fav_count = row.fav if row.fav else 0.0
      cart_count = row.cart if row.cart else 0.0
      buy_count = row.buy if row.buy else 0.0
      
      # 打分标准
      pv_score = 0.2*pv_count if pv_count<=20 else 4.0
      fav_score = 0.4*fav_count if fav_count<=20 else 8.0
      cart_score = 0.6*cart_count if cart_count<=20 else 12.0
      buy_score = 1.0*buy_count if buy_count<=20 else 20.0
      rating = pv_score + fav_score + cart_score + buy_score
      
      return row.userId, row.cateId, rating
  
  
  cate_rating_df = count_cate_df.rdd.map(process_row).toDF(["userId", "cateId", "rating"])
  cate_rating_df.show(5)
  ```

  ```python
  +-------+------+------+
  | userId|cateId|rating|
  +-------+------+------+
  | 754250|  6339|   0.2|
  |1053070|  4505|   0.8|
  | 148496|  6407|   0.2|
  | 184539|  4505|   1.0|
  |  96267|  4267|   0.2|
  +-------+------+------+
  only showing top 5 rows
  ```

  ②建模并预测

  ```python
  # 使用交替最小二乘法来更新P,Q矩阵
  from pyspark.ml.recommendation import ALS
      
  als = ALS(userCol="userId", itemCol="cateId", ratingCol="rating"
            , rank=10  # 隐含特征维度
           )  
  
  model = als.fit(cate_rating_df)
  model.save("data/ALS_Cate_model.obj")
  ```

  ```python
  # 为商品进行推荐
  model.recommendForAllItems(3).show(5, truncate=False)
  ```

  ```python
  +------+----------------------------------------------------------------+
  |cateId|recommendations                                                 |
  +------+----------------------------------------------------------------+
  |5300  |[[177286, 1.5867927], [175349, 1.5658401], [997962, 1.5140524]] |
  |7240  |[[615249, 5.3005347], [852025, 4.3769593], [530895, 3.9891386]] |
  |7340  |[[318626, 3.7813935], [1097591, 3.5653138], [77638, 3.1031678]] |
  |1591  |[[145444, 1.6491184], [775694, 0.91614044], [729248, 0.9116045]]|
  |10362 |[[274493, 3.0459692], [177286, 2.789281], [1031701, 2.7604806]] |
  +------+----------------------------------------------------------------+
  ```

  ```python
  # 为用户进行推荐(按照推荐评分降序号排列)
  model.recommendForAllUsers(3).show(5, truncate=False)
  ```

  ```python
  +------+--------------------------------------------------------------+
  |userId|recommendations                                               |
  +------+--------------------------------------------------------------+
  |471   |[[4826, 0.18844897], [7653, 0.16329545], [7619, 0.16224438]]  |
  |2142  |[[11831, 0.44443724], [6818, 0.27778324], [8971, 0.24943218]] |
  |2659  |[[11417, 0.22813325], [11486, 0.21443768], [6251, 0.18582432]]|
  |18979 |[[7393, 0.4638352], [609, 0.45257434], [5232, 0.44833192]]    |
  |28088 |[[5232, 0.3899253], [11417, 0.33631915], [9984, 0.23813573]]  |
  +------+--------------------------------------------------------------+
  only showing top 5 rows
  ```

  ③用户隐语义矩阵

  ```python
  # 用户隐语义矩阵
  model.userFactors.show(5, truncate=False)
  ```

  ```python
  +----+--------------------------------------------------------------------------------------------------------------------------------------+
  |id  |features                                                                                                                              |
  +----+--------------------------------------------------------------------------------------------------------------------------------------+
  |90  |[0.017561475, 0.045810036, -0.031669527, 0.041597474, 0.025158156, 0.03686702, 0.010029329, 0.18272054, 0.06516253, 0.0023921921]     |
  |870 |[0.14912017, 0.06403826, -0.011565263, -0.054896966, -0.025673112, 0.05948214, 0.00963695, -0.031872895, -0.015582311, -0.05498932]   |
  |1770|[0.015652183, 0.04841099, 0.0673799, 0.017094497, 0.069930434, 0.064016454, -0.08339966, -0.052054748, 0.068244345, 0.11521981]       |
  |1780|[-0.09248124, -0.021973772, 0.0028592395, 0.035712898, -0.066225946, 0.12873639, -0.05793281, -0.049343813, 0.010344725, -0.098594226]|
  |1900|[-0.23048756, 0.09836291, -0.15108949, -0.07170922, -0.16445853, -0.08974109, 0.2773702, -0.0956187, -0.39774004, 0.14901641]         |
  +----+--------------------------------------------------------------------------------------------------------------------------------------+
  only showing top 5 rows
  ```

  ④预测指定用户

  ```python
  dataset = spark.createDataFrame([[471],[28088],[18979]])
  dataset = dataset.withColumnRenamed("_1", "userId")  # columns名称要与用户评分表保持一致
  model.recommendForUserSubset(dataset, 4).show(truncate=False)
  ```

  ```python
  +------+--------------------------------------------------------------------------------+
  |userId|recommendations                                                                 |
  +------+--------------------------------------------------------------------------------+
  |28088 |[[5232, 0.3899253], [11417, 0.33631915], [9984, 0.23813573], [7967, 0.22711103]]|
  |471   |[[4826, 0.18844897], [7653, 0.16329545], [7619, 0.16224438], [4286, 0.13812888]]|
  |18979 |[[7393, 0.4638352], [609, 0.45257434], [5232, 0.44833192], [10998, 0.4183498]]  |
  +------+--------------------------------------------------------------------------------+
  ```

- **构建ALS模型（基于品牌类型）**

  ![](./imgs/LFM矩阵分解图解.png)

  ①计算rating

  ```python
  def process_row(row):
      """获取用户品牌评分数据"""
      
      # 获取指定次数
      pv_count = row.pv if row.pv else 0.0
      fav_count = row.fav if row.fav else 0.0
      cart_count = row.cart if row.cart else 0.0
      buy_count = row.buy if row.buy else 0.0
      
      # 打分标准
      pv_score = 0.2*pv_count if pv_count<=20 else 4.0
      fav_score = 0.4*fav_count if fav_count<=20 else 8.0
      cart_score = 0.6*cart_count if cart_count<=20 else 12.0
      buy_score = 1.0*buy_count if buy_count<=20 else 20.0
      
      rating = pv_score + fav_score + cart_score + buy_score
      
      return row.userId, row.brandId, rating
  
  
  count_brand_df = count_brand_df.rdd.map(process_row).toDF(["userId", "brandId", "rating"])
  ```

  ②建模

  ```python
  # 使用交替最小二乘法来更新P,Q矩阵
  from pyspark.ml.recommendation import ALS
  
  als = ALS(userCol="userId", itemCol="brandId", ratingCol="rating", rank=10)
  model = als.fit(count_brand_df)
  ```

  ③预测

  ```python
  model.recommendForAllUsers(2).show(5)
  ```

  ```python
  +------+--------------------+
  |userId|     recommendations|
  +------+--------------------+
  |   471|[[91086, 0.288672...|
  |  2142|[[359751, 0.41074...|
  |  2659|[[312873, 0.53953...|
  | 18979|[[5928, 1.3026588...|
  | 28088|[[91086, 0.667446...|
  +------+--------------------+
  only showing top 5 rows
  ```

### 参考

[推荐系统](https://www.jiqizhixin.com/graph/technologies/6ca1ea2d-6bca-45b7-9c93-725d288739c3)

[黑马python5.0](http://www.itheima.com/special/pythonzly/)

[推荐系统实战（4）——基于模型的协同过滤算法（隐语义模型LFM）（代码实现](https://blog.csdn.net/a1272899331/article/details/105159964?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf)































































