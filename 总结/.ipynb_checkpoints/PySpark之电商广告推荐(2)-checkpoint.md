## 数据预处理

### raw_sample表

- **表描述**

  淘宝网站中随机抽样了114万用户8天内的广告展示/点击日志（2600万条记录），构成原始的样本骨架。 字段说明如下：

  1. user_id：脱敏过的用户ID；
  2. adgroup_id：脱敏过的广告单元ID；
  3. time_stamp：时间戳；
  4. pid：资源位；
  5. noclk：为1代表没有点击；为0代表点击；
  6. clk：为0代表没有点击；为1代表点击；

- **数据读取并分析**

  ```python
  # 淘宝网站中随机抽样了114万用户8天内的广告展示/点击日志（2600万条记录）构成原始的样本数据
  df = spark.read.csv("data/raw_sample.csv", header=True)
  df.show(5)
  ```

  ```pythoon
  +------+----------+----------+-----------+------+---+
  |  user|time_stamp|adgroup_id|        pid|nonclk|clk|
  +------+----------+----------+-----------+------+---+
  |581738|1494137644|         1|430548_1007|     1|  0|
  |449818|1494638778|         3|430548_1007|     1|  0|
  |914836|1494650879|         4|430548_1007|     1|  0|
  |914836|1494651029|         5|430548_1007|     1|  0|
  |399907|1494302958|         8|430548_1007|     1|  0|
  +------+----------+----------+-----------+------+---+
  only showing top 5 rows
  ```

  ```python
  print("总样本数目：", df.count())
  print("adgroup_id数目：", df.groupBy("adgroup_id").count().count())
  print("广告示位置：", df.groupBy("pid").count().collect())  # 可以考虑热编码onehot
  print("用户的点击情况：", df.groupBy("nonclk").count().collect())
  ```

  ```python
  总样本数目： 26557961
  adgroup_id数目： 846811
  广告示位置： [Row(pid='430548_1007', count=16472898), Row(pid='430539_1007', count=10085063)]
  用户的点击情况： [Row(nonclk='0', count=1366056), Row(nonclk='1', count=25191905)]
  ```

- **更改数据类型**

  ```python
  from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType
  
  # 打印df结构信息
  df.printSchema()   
  # 更改df表结构：更改列类型和列名称
  raw_sample_df = df.\
      withColumn("user", df.user.cast(IntegerType())).withColumnRenamed("user", "userId").\
      withColumn("time_stamp", df.time_stamp.cast(LongType())).withColumnRenamed("time_stamp", "timestamp").\
      withColumn("adgroup_id", df.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId").\
      withColumn("pid", df.pid.cast(StringType())).\
      withColumn("nonclk", df.nonclk.cast(IntegerType())).\
      withColumn("clk", df.clk.cast(IntegerType()))
  
  raw_sample_df.printSchema()
  ```

  ```python
  root
   |-- user: string (nullable = true)
   |-- time_stamp: string (nullable = true)
   |-- adgroup_id: string (nullable = true)
   |-- pid: string (nullable = true)
   |-- nonclk: string (nullable = true)
   |-- clk: string (nullable = true)
  
  root
   |-- userId: integer (nullable = true)
   |-- timestamp: long (nullable = true)
   |-- adgroupId: integer (nullable = true)
   |-- pid: string (nullable = true)
   |-- nonclk: integer (nullable = true)
   |-- clk: integer (nullable = true)
  ```

- **"pid"字段热编码**

  Spark中使用热独编码
  
  - 注意：**热编码只能对字符串类型的列数据进行处理**
  
   [StringIndexer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer)：对指定字符串列数据进行特征处理，如将性别数据“男”、“女”转化为0和1
  
   [OneHotEncoder](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder)：对特征列数据，进行热编码，通常需结合StringIndexer一起使用
  
   [Pipeline](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=pipeline#pyspark.ml.Pipeline)：让数据按顺序依次被处理，将前一次的处理结果作为下一次的输入

  ```python
  from pyspark.ml.feature import OneHotEncoder
  from pyspark.ml.feature import StringIndexer
  from pyspark.ml import Pipeline
  
  stringindexer = StringIndexer(inputCol="pid", outputCol="pid_feature")
  onehot = OneHotEncoder(inputCol="pid_feature", outputCol="pid_value", dropLast=False)
  pipeline = Pipeline(stages=[stringindexer, onehot])
  pipeline_model = pipeline.fit(df)
  new_df = pipeline_model.transform(df)
  # 查看onehot编码的结果
  new_df.groupBy("pid_value").count().show()
  ```

  ```python
  +-------------+--------+
  |    pid_value|   count|
  +-------------+--------+
  |(2,[0],[1.0])|16472898|
  |(2,[1],[1.0])|10085063|
  +-------------+--------+
  ```

- **时间戳字段分析**

  ```python
  from datetime import datetime
  
  # 查看时间
  new_df.sort("time_stamp", ascending=False).show()
  # 留取最后一天为测试集, 前七天为训练集
  print("第八天:", datetime.fromtimestamp(1494691186))
  print("第七天分割点:", datetime.fromtimestamp(1494691186-24*60*60))
  ```

  ```python
  第八天: 2017-05-13 23:59:46
  第七天分割点: 2017-05-12 23:59:46
  ```

  ```python
  train_sample = new_df.filter(new_df.time_stamp<=(1494691186-24*60*60))
  test_sample = new_df.filter(new_df.time_stamp>(1494691186-24*60*60))
  # 所占分数
  train_sample.count(), test_sample.count()
  ```

  ```python
  (23249291, 3308670)
  ```

### ad_feature表

- **表描述**

  本数据集涵盖了raw_sample中全部广告的基本信息(约80万条目)。字段说明如下：

  1. adgroup_id：脱敏过的广告ID；
  2. cate_id：脱敏过的商品类目ID；
  3. campaign_id：脱敏过的广告计划ID；
  4. customer_id: 脱敏过的广告主ID；
  5. brand_id：脱敏过的品牌ID；
  6. price: 宝贝的价格

  其中一个广告ID对应一个商品（宝贝），一个宝贝属于一个类目，一个宝贝属于一个品牌。

- **数据读取并分析**

  ```python
  # 广告信息表
  adf = spark.read.csv("data/ad_feature.csv", header=True)
  adf.show(5)
  ```

  ```python
  +----------+-------+-----------+--------+------+-----+
  |adgroup_id|cate_id|campaign_id|customer| brand|price|
  +----------+-------+-----------+--------+------+-----+
  |     63133|   6406|      83237|       1| 95471|170.0|
  |    313401|   6406|      83237|       1| 87331|199.0|
  |    248909|    392|      83237|       1| 32233| 38.0|
  |    208458|    392|      83237|       1|174374|139.0|
  |    110847|   7211|     135256|       2|145952|32.99|
  +----------+-------+-----------+--------+------+-----+
  only showing top 5 rows
  ```

  ```python
  adf.printSchema(), adf.count()
  ```

  ```python
  root
   |-- adgroup_id: string (nullable = true)
   |-- cate_id: string (nullable = true)
   |-- campaign_id: string (nullable = true)
   |-- customer: string (nullable = true)
   |-- brand: string (nullable = true)
   |-- price: string (nullable = true)
  
  (None, 846811)
  ```

- **更改字段类型**

  ```python
  # 首先填补NULL值为-1后，并修改其对应字段的数据类型
  adf = adf.replace("NULL", "-1")
  
  # 修改数据类型
  ad_feature_df = adf.withColumn("adgroup_id", adf.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupID").\
      withColumn("cate_id", adf.cate_id.cast(IntegerType())).withColumnRenamed("cate_id", "cateId").\
      withColumn("campaign_id", adf.campaign_id.cast(IntegerType())).withColumnRenamed("campaign_id", "campaignId").\
      withColumn("customer", adf.customer.cast(IntegerType())).withColumnRenamed("customer", "customerId").\
      withColumn("brand", adf.brand.cast(IntegerType())).withColumnRenamed("brand", "brandId").\
      withColumn("price", adf.price.cast(FloatType()))
  
  ad_feature_df.printSchema()
  ```

  ```python
  root
   |-- adgroupID: integer (nullable = true)
   |-- cateId: integer (nullable = true)
   |-- campaignId: integer (nullable = true)
   |-- customerId: integer (nullable = true)
   |-- brandId: integer (nullable = true)
   |-- price: float (nullable = true)
  ```

- 统计

  ```python
  # 基本数据指标统计
  print("总广告条数：",df.count())   # 数据条数
  _1 = ad_feature_df.groupBy("cateId").count().count()
  print("cateId数值个数：", _1)
  _2 = ad_feature_df.groupBy("campaignId").count().count()
  print("campaignId数值个数：", _2)
  _3 = ad_feature_df.groupBy("customerId").count().count()
  print("customerId数值个数：", _3)
  _4 = ad_feature_df.groupBy("brandId").count().count()
  print("brandId数值个数：", _4)
  print("价格高于1w的条目个数:", ad_feature_df.filter(ad_feature_df.price > 10000).count())
  print("价格低于1w的条目个数:", ad_feature_df.filter(ad_feature_df.price <= 10000).count())
  
  ad_feature_df.sort("price").show()
  ad_feature_df.sort("price", ascending=False).show()
  ad_feature_df.describe().show()
  ```

  ```python
  总广告条数： 26557961
  cateId数值个数： 6769
  campaignId数值个数： 423436
  customerId数值个数： 255875
  brandId数值个数： 99815
  价格高于1w的条目个数: 6527
  价格低于1w的条目个数: 840284
  +---------+------+----------+----------+-------+-----+
  |adgroupID|cateId|campaignId|customerId|brandId|price|
  +---------+------+----------+----------+-------+-----+
  |    92241|  6130|     72781|    149714|     -1| 0.01|
  |   149570|  7043|    126746|    176076|     -1| 0.01|
  |    71678|  9866|    124203|     91492|  63885| 0.01|
  |   345870|  9995|    179595|    191036|  79971| 0.01|
  |    41925|  7032|     85373|    114532|     -1| 0.01|
  |    88975|  9996|    198424|    182415|     -1| 0.01|
  |   485749|  9970|    352666|    140520|     -1| 0.01|
  |   494084|  9969|    349384|    154919|     -1| 0.01|
  |    49911|  7032|    129079|    172334|     -1| 0.01|
  |    42055|  9994|     43866|    113068| 123242| 0.01|
  |   692990|  6018|    353223|    223320|     -1| 0.01|
  |   348342|  8999|    296966|    158809| 113555| 0.01|
  |   288172|  9995|    314179|    230326| 399440| 0.01|
  |   620285|  7043|    365821|      1960| 188191| 0.01|
  |   174248|  8999|    184344|    196777| 113555| 0.01|
  |   290675|  4824|    315371|    240984|     -1| 0.01|
  |   598024|  9970|     22467|     59048|  17554| 0.01|
  |   517587|  1847|    352238|    158227| 188592| 0.01|
  |   182565|  5375|    274375|     16356|     -1| 0.01|
  |   169988| 10539|    238823|    221154| 211816| 0.01|
  +---------+------+----------+----------+-------+-----+
  only showing top 20 rows
  
  +---------+------+----------+----------+-------+-----------+
  |adgroupID|cateId|campaignId|customerId|brandId|      price|
  +---------+------+----------+----------+-------+-----------+
  |   179746|  1093|    270027|    102509| 405447|      1.0E8|
  |   658722|  1093|    218101|    207754|     -1|      1.0E8|
  |   443295|  1093|     44251|    102509| 300681|      1.0E8|
  |   468220|  1093|    270719|    207754|     -1|      1.0E8|
  |   243384|   685|    218918|     31239| 278301|      1.0E8|
  |    31899|   685|    218918|     31239| 278301|      1.0E8|
  |   554311|  1093|    266086|    207754|     -1|      1.0E8|
  |   513942|   745|      8401|     86243|     -1|8.8888888E7|
  |   201060|   745|      8401|     86243|     -1|5.5555556E7|
  |   289563|   685|     37665|    120847| 278301|      1.5E7|
  |    35156|   527|    417722|     72273| 278301|      1.0E7|
  |    33756|   527|    416333|     70894|     -1|  9900000.0|
  |   335495|   739|    170121|    148946| 326126|  9600000.0|
  |   218306|   206|    162394|      4339| 221720|  8888888.0|
  |   213567|  7213|    239302|    205612| 406125|  5888888.0|
  |   375920|   527|    217512|    148946| 326126|  4760000.0|
  |   262215|   527|    132721|     11947| 417898|  3980000.0|
  |   154623|   739|    170121|    148946| 326126|  3900000.0|
  |   152414|   739|    170121|    148946| 326126|  3900000.0|
  |   448651|   527|    422260|     41289| 209959|  3800000.0|
  +---------+------+----------+----------+-------+-----------+
  only showing top 20 rows
  
  +-------+-----------------+------------------+------------------+------------------+------------------+------------------+
  |summary|        adgroupID|            cateId|        campaignId|        customerId|           brandId|             price|
  +-------+-----------------+------------------+------------------+------------------+------------------+------------------+
  |  count|           846811|            846811|            846811|            846811|            846811|            846811|
  |   mean|         423406.0| 5868.593464185043|206552.60428005777|113180.40600559038|162566.00186464275| 1838.867108130995|
  | stddev|244453.4237388929|2705.1712033181752|125192.34090758236| 73435.83494972308| 152482.7386634471|310887.70017026004|
  |    min|                1|                 1|                 1|                 1|                -1|              0.01|
  |    max|           846811|             12960|            423436|            255875|            461497|             1.0E8|
  +-------+-----------------+------------------+------------------+------------------+------------------+------------------+
  ```

### user_profile表

- **表描述**

  用户基本信息表user_profile

  本数据集涵盖了raw_sample中全部用户的基本信息(约100多万用户)。字段说明如下：

  1. userid：脱敏过的用户ID；
  2. cms_segid：微群ID；
  3. cms_group_id：cms_group_id；
  4. final_gender_code：性别 1:男,2:女；
  5. age_level：年龄层次； 1234
  6. pvalue_level：消费档次，1:低档，2:中档，3:高档；
  7. shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
  8. occupation：是否大学生 ，1:是,0:否
  9. new_user_class_level：城市层级

- **数据读取并分析**

  ```python
  # 用户信息表
  upf = spark.read.csv("data/user_profile.csv", header=True)
  upf.show(5)
  ```

  ```python
  +------+---------+------------+-----------------+---------+------------+--------------+----------+---------------------+
  |userid|cms_segid|cms_group_id|final_gender_code|age_level|pvalue_level|shopping_level|occupation|new_user_class_level |
  +------+---------+------------+-----------------+---------+------------+--------------+----------+---------------------+
  |   234|        0|           5|                2|        5|        null|             3|         0|                    3|
  |   523|        5|           2|                2|        2|           1|             3|         1|                    2|
  |   612|        0|           8|                1|        2|           2|             3|         0|                 null|
  |  1670|        0|           4|                2|        4|        null|             1|         0|                 null|
  |  2545|        0|          10|                1|        4|        null|             3|         0|                 null|
  +------+---------+------------+-----------------+---------+------------+--------------+----------+---------------------+
  only showing top 5 rows
  ```

- **更改字段类型**

  ```python
  from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType
  
  # 构建表结构schema对象
  schema = StructType([
      StructField("userId", IntegerType()),
      StructField("cms_segid", IntegerType()),
      StructField("cms_group_id", IntegerType()),
      StructField("final_gender_code", IntegerType()),
      StructField("age_level", IntegerType()),
      StructField("pvalue_level", IntegerType()),
      StructField("shopping_level", IntegerType()),
      StructField("occupation", IntegerType()),
      StructField("new_user_class_level", IntegerType())
  ])
  
  # 利用schema从hdfs加载
  user_profile_df = spark.read.csv("./data/user_profile.csv", header=True, schema=schema)
  user_profile_df.printSchema()
  ```

  ```python
  root
   |-- userId: integer (nullable = true)
   |-- cms_segid: integer (nullable = true)
   |-- cms_group_id: integer (nullable = true)
   |-- final_gender_code: integer (nullable = true)
   |-- age_level: integer (nullable = true)
   |-- pvalue_level: integer (nullable = true)
   |-- shopping_level: integer (nullable = true)
   |-- occupation: integer (nullable = true)
   |-- new_user_class_level: integer (nullable = true)
  ```

- **统计**

  ```python
  print("分类特征值个数情况: ")
  print("cms_segid: ", user_profile_df.groupBy("cms_segid").count().count())
  print("cms_group_id: ", user_profile_df.groupBy("cms_group_id").count().count())
  print("final_gender_code: ", user_profile_df.groupBy("final_gender_code").count().count())
  print("age_level: ", user_profile_df.groupBy("age_level").count().count())
  print("shopping_level: ", user_profile_df.groupBy("shopping_level").count().count())
  print("occupation: ", user_profile_df.groupBy("occupation").count().count())
  ```

  ```python
  分类特征值个数情况: 
  cms_segid:  97  # 特征取值较多，不宜升高维度
  cms_group_id:  13
  final_gender_code:  2
  age_level:  7  # 七档消费档次
  shopping_level:  3  # 三档购物层次
  occupation:  2
  ```

- **缺失值**

  表中的pvalue_level和new_user_class_level字段含有部分缺失值，需要对缺失值进行填补，处理步骤一般如下：

  - 缺失值处理

    - 注意，一般情况下：

      - 缺失率低于10%：可直接进行相应的填充，如默认值、均值、算法拟合等等；
      - 高于10%：往往会考虑舍弃该特征
      - <font color="blue">特征处理，如1维转多维</font>

      但根据我们的经验，我们的广告推荐其实和用户的消费水平、用户所在城市等级都有比较大的关联，因此在这里pvalue_level、new_user_class_level都是比较重要的特征，我们不考虑舍弃

  - 缺失值处理方案：

    - 填充方案：结合用户的其他特征值，利用随机森林算法进行预测；但产生了大量人为构建的数据，一定程度上增加了数据的噪音
    - 把变量映射到高维空间：如pvalue_level的1维数据，转换成是否1、是否2、是否3、是否缺失的4维数据；这样保证了所有原始数据不变，同时能提高精确度，<font color="red">但这样会导致数据变得比较稀疏，如果样本量很小，反而会导致样本效果较差，因此也不能滥用</font>

  - 填充方案

    - 利用随机森林对pvalue_level的缺失值进行预测

  **缺失情况**

  ```python
print("含缺失值的特征情况: ")
  user_profile_df.groupBy("pvalue_level").count().show()
user_profile_df.groupBy("new_user_class_level").count().show()
  ```
  
  ```python
  含缺失值的特征情况: 
  +------------+------+
|pvalue_level| count|
  +------------+------+
  |        null|575917|
  |           1|154436|
  |           3| 37759|
  |           2|293656|
  +------------+------+
  
  +--------------------+------+
  |new_user_class_level| count|
  +--------------------+------+
  |                null|344920|
  |                   1| 80548|
  |                   3|173047|
  |                   4|138833|
  |                   2|324420|
  ```
  
  **缺失比率**
  
  ```python
t_count = user_profile_df.count()
  
  ```

pl_na_count = t_count - user_profile_df.dropna(subset=["pvalue_level"]).count()
  print("pvalue_level空值个数：", pl_na_count, "空占比%0.2f%%"%(pl_na_count/t_count))

  # 此时缺失值比重较大, 但由于其自身对最终的预测有决定性的作用,所以可以考虑进行填补
  nul_na_count = t_count - user_profile_df.dropna(subset=["new_user_class_level"]).count()
  print('nul_na_count空值个数：', nul_na_count, "空占比%0.2f%%"%(nul_na_count/t_count))
  ```
  
  ```python
  pvalue_level空值个数： 575917 空占比0.54%
  nul_na_count空值个数： 344920 空占比0.32%
  ```

- **缺失值填补**：<font color="blue">随机森林(使用mllib,则需要转为与之对应的rdd格式数据类型)</font>
  
  **构建训练集**（相对应字段不为空）
  
  ```python
  from pyspark.mllib.linalg import SparseVector
  from pyspark.mllib.regression import LabeledPoint
    
    # 构建随机森林填补的训练集和测试集
    # 保证标签从0开始
    train_data = user_profile_df.dropna(subset=["pvalue_level"])\
                .rdd.map(lambda r: LabeledPoint(r.pvalue_level-1, 
                                                [r.cms_segid, r.cms_group_id, r.final_gender_code
                                                 , r.age_level, r.shopping_level, r.occupation])
                                                )
    
    # 对城市等级缺失值进行填补
    # 选出new_user_class_level全部的
    train_data2 = user_profile_df.dropna(subset=["new_user_class_level"]).rdd.map(
        lambda r:LabeledPoint(r.new_user_class_level - 1, [r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation])
    )
  ```
  
    **建模**
  
    ```python
    # model模型
    from pyspark.mllib.tree import RandomForest  # RDD数据类型
    model = RandomForest().trainClassifier(train_data, numClasses=3, categoricalFeaturesInfo={}, numTrees=5)
    model2 = RandomForest().trainClassifier(train_data2, 4, {}, 5)
    # 单个样本预测
    model.predict([5.0,2.0,2.0,2.0,3.0,1.0])
    ```
  
    **构建测试集**（相对应字段为空）
  
    ```python
    # 构建测试集
    pl_na_df = user_profile_df.na.fill(-1).where("pvalue_level=-1")
    nul_na_df = user_profile_df.na.fill(-1).where("new_user_class_level=-1")
    
    # 转换为指定数据类型
    def row(r):
        return r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation
    
    rdd2 = nul_na_df.rdd.map(row)
    predicts2 = model.predict(rdd2)
    
    rdd = pl_na_df.rdd.map(row) 
    # 预测缺失值的数据
    predicts = model.predict(rdd)
    print(predicts.take(5))
    predicts.count()
    ```
  
    ```python
    [1.0, 1.0, 1.0, 1.0, 1.0]
    575917
    ```
  
    **填补字段缺失值**
  
    ```python
    # label 加１,转为pandas进行数据处理
    import numpy as np
    
    temp = predicts.map(lambda x:int(x)).collect()
    pdf = pl_na_df.toPandas()  # 选择需要填补的空值
    pdf["pvalue_level"] = np.array(temp)+1
    
    # 数据填补完成
    new_user_profile_df = user_profile_df.dropna(subset=["pvalue_level"]).unionAll(spark.createDataFrame(pdf, schema=schema))
    new_user_profile_df.show(5)
    ```
  
    ```python
    +------+---------+------------+-----------------+---------+------------+--------------+----------+--------------------+
    |userId|cms_segid|cms_group_id|final_gender_code|age_level|pvalue_level|shopping_level|occupation|new_user_class_level|
    +------+---------+------------+-----------------+---------+------------+--------------+----------+--------------------+
    |   523|        5|           2|                2|        2|           1|             3|         1|                   2|
    |   612|        0|           8|                1|        2|           2|             3|         0|                null|
    |  3644|       49|           6|                2|        6|           2|             3|         0|                   2|
    |  5777|       44|           5|                2|        5|           2|             3|         0|                   2|
    |  6355|        2|           1|                2|        1|           1|             3|         0|                   4|
    +------+---------+------------+-----------------+---------+------------+--------------+----------+--------------------+
    only showing top 5 row
    ```
  
- **缺失值填补**：<font color="blue">低维转高纬方式</font>

  目的：我们接下来采用将变量映射到高维空间的方法来处理数据，即将缺失项也当做一个单独的特征来对待，保证数据的原始性，由于该思想正好和热独编码实现方法一样，因此这里直接使用热独编码方式处理数据

  ```python
  from pyspark.ml.feature import OneHotEncoder
  from pyspark.ml.feature import StringIndexer
  from pyspark.ml import Pipeline
  from pyspark.sql.types import StringTyp
  
  # 需要填补空值后才能进行类型字段类型装换
  user_profile_df = user_profile_df.na.fill(-1)
  
  
  # 转变onehot编码,必须将相对应字段的数据进行填补字符串类型
  user_profile_df = user_profile_df.withColumn("pvalue_level", user_profile_df.pvalue_level.cast(StringType()))\
                                  .withColumn("new_user_class_level", user_profile_df.new_user_class_level.cast(StringType()))
  
  # 1. pvalue_level字段onehot
  # onehot编码一般流程　pvalue_level字段
  stringindex = StringIndexer(inputCol="pvalue_level", outputCol="pl_onehot_feature")
  encoder = OneHotEncoder(inputCol="pl_onehot_feature", outputCol="pl_onehot_value", dropLast=False)
  pipeline = Pipeline(stages=[stringindex, encoder])
  pipeline_fit = pipeline.fit(user_profile_df)
  user_profile_df2 = pipeline_fit.transform(user_profile_df)
  
  # 2.new_user_class_level字段onehot
  stringindexer = StringIndexer(inputCol='new_user_class_level', outputCol='nucl_onehot_feature')
  encoder = OneHotEncoder(dropLast=False, inputCol='nucl_onehot_feature', outputCol='nucl_onehot_value')
  pipeline = Pipeline(stages=[stringindexer, encoder])
  pipeline_fit = pipeline.fit(user_profile_df2)
  user_profile_df3 = pipeline_fit.transform(user_profile_df2)
  user_profile_df3.show(5, truncate=False)
  ```

  **VectorAssembler**(ml指定格式类型)

  ```python
  feature_df = VectorAssembler().setInputCols(["age_level", "pl_onehot_value", "nucl_onehot_value"]).setOutputCol("feature").transform(user_profile_df3)
  feature_df.select("feature").show(5, truncate=False)
  ```

  ```python
  +--------------------------+
  |feature                   |
  +--------------------------+
  |(10,[0,1,7],[5.0,1.0,1.0])|
  |(10,[0,3,6],[2.0,1.0,1.0])|
  |(10,[0,2,5],[2.0,1.0,1.0])|
  |(10,[0,1,5],[4.0,1.0,1.0])|
  |(10,[0,1,5],[4.0,1.0,1.0])|
  +--------------------------+
  only showing top 5 rows
  ```


### 参考

[推荐系统](https://www.jiqizhixin.com/graph/technologies/6ca1ea2d-6bca-45b7-9c93-725d288739c3)

[黑马python5.0](http://www.itheima.com/special/pythonzly/)

[推荐系统（一）：个性化电商广告推荐系统介绍、数据集介绍、项目效果展示、项目实现分析、点击率预测(CTR--Click-Through-Rate)概念](https://blog.csdn.net/qq_35456045/article/details/104881691?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159935430019724839860552%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=159935430019724839860552&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~rank_business_v1-6-104881691.ecpm_v3_rank_business_v1&utm_term=%E4%B8%AA%E6%80%A7%E5%8C%96%E7%94%B5%E5%95%86%E6%8E%A8%E8%8D%90&spm=1018.2118.3001.4187)