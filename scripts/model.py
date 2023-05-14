import math
import time
import itertools
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, Transformer
from pyspark.sql.window import Window
from pyspark.ml.evaluation import Evaluator
from pyspark.sql.types import StringType, FloatType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector, UnivariateFeatureSelector
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, Evaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from tqdm import tqdm
from pyspark import since, keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.common import _py2java
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasSeed
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.ml.util import *
from pyspark.ml.wrapper import JavaParams
from pyspark.sql.functions import rand
from functools import reduce


# Transformation
class CustomTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    input_col = Param(Params._dummy(), "input_col",
                      "input column name.", typeConverter=TypeConverters.toString)
    output_col = Param(Params._dummy(), "output_col",
                       "output column name.", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, input_col="input", output_col="output"):
        super(CustomTransformer, self).__init__()
        self._setDefault(input_col=None, output_col=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, input_col="input", output_col="output"):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_input_col(self):
        return self.getOrDefault(self.input_col)

    def get_output_col(self):
        return self.getOrDefault(self.output_col)

    def _transform(self, df: DataFrame):
        input_col = self.get_input_col()
        output_col = self.get_output_col()

        if input_col == 'trans_month':
            val = 12
        elif input_col == 'trans_day':
            val = 7
        elif input_col == 'trans_hour':
            val = 24
        else:
            val = 1

        return df.withColumn(output_col + '_sin', F.sin(2*math.pi*F.col(input_col)/val))\
            .withColumn(output_col + '_cos', F.cos(2*math.pi*F.col(input_col)/val)).drop(input_col)


"""TrainTest split"""
# split dataframes between 0s and 1s


def stratified_split(df, label, test_size=0.2, seed=1):
    zeros = df.filter(df[label] == 0)
    ones = df.filter(df[label] == 1)
    # split datasets into training and testing
    train0, test0 = zeros.randomSplit([1.0-test_size, test_size], seed=seed)
    train1, test1 = ones.randomSplit([1.0-test_size, test_size], seed=seed)
    # stack datasets back together
    train = train1.union(train0)
    test = test1.union(test0)
    # SHUFFLE!
    train = train.select("*").orderBy(F.rand(seed=seed))
    test = test.select("*").orderBy(F.rand(seed=seed))
    return train, test


# function for evaluating the model
def evaluate(predictions, label):
    area_pr = BinaryClassificationEvaluator(
        rawPredictionCol='prediction', labelCol=label, metricName='areaUnderPR')
    print("Area under PR curve: ", area_pr.evaluate(predictions))
    print("Confusion matrix")
    conf_matrix = predictions.groupBy('is_fraud', 'prediction').count()
    conf_matrix.show()
    # Calculate the elements of the confusion matrix
    TN = predictions.filter(f'prediction = 0 AND {label} = prediction').count()
    TP = predictions.filter(f'prediction = 1 AND {label} = prediction').count()
    FN = predictions.filter(
        f'prediction = 0 AND {label} <> prediction').count()
    FP = predictions.filter(
        f'prediction = 1 AND {label} <> prediction').count()
    # calculate accuracy, precision, recall, and F1-score
    accuracy = (TN + TP) / (TN + TP + FN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F = 2 * (precision*recall) / (precision + recall + 1e-8)
    print("threshold 0.5: ")
    print('precision: %0.3f' % precision)
    print('recall: %0.3f' % recall)
    print('accuracy: %0.3f' % accuracy)
    print('F1 score: %0.3f' % F)
    return area_pr, conf_matrix, accuracy, precision, recall, F


# function for Stratified K-Fold
class StratifiedCrossValidator(CrossValidator):
    def stratify_data(self, dataset):
        """
        Returns an array of dataframes with the same ratio of passes and failures.

        Currently only supports binary classification problems.
        """
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        nFolds = self.getOrDefault(self.numFolds)
        split_ratio = 1.0 / nFolds

        passes = dataset[dataset['is_fraud'] == 1]
        fails = dataset[dataset['is_fraud'] == 0]

        pass_splits = passes.randomSplit(
            [split_ratio for i in range(nFolds)], seed=1)
        fail_splits = fails.randomSplit(
            [split_ratio for i in range(nFolds)], seed=1)

        stratified_data = [pass_splits[i].unionAll(
            fail_splits[i]) for i in range(nFolds)]

        return stratified_data

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        metrics = [0.0] * numModels

        stratified_data = self.stratify_data(dataset)

        for i in tqdm(range(nFolds)):
            start = time.time()
            train_arr = [x for j, x in enumerate(stratified_data) if j != i]
            end = time.time()
            print(end-start)
            train = reduce((lambda x, y: x.unionAll(y)), train_arr)
            end2 = time.time()
            print(end2-end)
            validation = stratified_data[i]
            start = time.time()
            models = est.fit(train, epm)
            print(time.time()-start)

            for j in tqdm(range(numModels)):
                model = models[j]
                metric = eva.evaluate(model.transform(validation, epm[j]))
                metrics[j] += metric/nFolds

            if eva.isLargerBetter():
                bestIndex = np.argmax(metrics)
            else:
                bestIndex = np.argmin(metrics)

        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics))


# custom metric
class CostEvaluator(Evaluator):
    def __init__(self, predictionCol="prediction", labelCol="label"):
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):
        FN_data = dataset.filter(f'{self.predictionCol} = 0 AND {self.labelCol} <> {self.predictionCol}')\
            .select(F.col("amt"))
        if FN_data.count() > 0:
            FN = FN_data.agg({'amt': 'sum'}).collect()[0][0]
        else:
            FN = 0

        FP = dataset.filter(
            f'{self.predictionCol} = 1 AND {self.labelCol} <> {self.predictionCol}').count()
        fp_cost = 4
        return (FN + FP*fp_cost)

    def isLargerBetter(self):
        return False


spark = SparkSession.builder\
    .appName("BDT Project")\
    .master("local[*]")\
    .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")\
    .config("spark.sql.catalogImplementation", "hive")\
    .config("spark.sql.avro.compression.codec", "snappy")\
    .config("spark.jars", "file:///usr/hdp/current/hive-client/lib/hive-metastore-1.2.1000.2.6.5.0-292.jar,file:///usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar")\
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.0.3")\
    .config("spark.driver.memory", "1g")\
    .enableHiveSupport()\
    .getOrCreate()

"""## Reading Tables"""
transactions_df = spark.read.format(
    "avro").table('projectdb.transactions_part')
cart_holder_df = spark.read.format("avro").table('projectdb.cart_holder_buck')
merchant_df = spark.read.format("avro").table('projectdb.merchant_buck')

# fixing seed
random.seed(1)

# Preprocessing
# Checking for null values


def check_null(df):
    num_null_cols = 0
    for c in df.columns:
        if df.filter(F.col(c).isNull()).count() > 0:
            num_null_cols += 1

    print("Num null columns: ", num_null_cols)
    return num_null_cols


check_null(merchant_df)
check_null(transactions_df)
check_null(merchant_df)

# Checking for duplicates


def check_duplicates(df, index_col):
    if df.count() > df.dropDuplicates([col for col in df.columns if col != index_col]).count():
        print('Data has duplicates')
    else:
        print("No duplicates")


check_duplicates(transactions_df, 'index')
check_duplicates(merchant_df, 'merchant_id')
check_duplicates(cart_holder_df, 'cart_holder_id')

"""Joining all tables"""

full_df = transactions_df.join(merchant_df,
                               transactions_df.merchant_id == merchant_df.merchant_id,
                               "inner").join(cart_holder_df,
                                             transactions_df.cart_holder_id == cart_holder_df.cart_holder_id,
                                             "inner").drop('merchant_id', 'cart_holder_id')

full_df.show(3)


"""## Feature engineering """

full_df = full_df.withColumn("age", F.floor(F.months_between(F.from_unixtime(F.col("trans_date_trans_time")),
                                                             F.from_unixtime(F.col("dob")))/F.lit(12)))

# add time features
full_df = full_df.withColumn("trans_month", F.month(
    F.from_unixtime(F.col("trans_date_trans_time"))))
full_df = full_df.withColumn("trans_hour", F.hour(
    F.from_unixtime(F.col("trans_date_trans_time"))))
full_df = full_df.withColumn("trans_day", F.dayofweek(
    F.from_unixtime(F.col("trans_date_trans_time"))))

# distance from customer location to merchant location in degrees latitude and degrees longitude
full_df = full_df.withColumn("lat_dist", F.abs(
    F.round(F.col("merch_lat")-F.col("lat"), 3)))
full_df = full_df.withColumn("long_dist", F.abs(
    F.round(F.col("merch_long")-F.col("long"), 3)))

full_df = full_df.withColumn("distance", F.sqrt(
    F.col("lat_dist") * 112 ** 2 + F.col("long_dist") * 112 ** 2))

Windowspec = Window.partitionBy("cc_num").orderBy("trans_date_trans_time")

full_df = full_df.withColumn(
    'recency', ((F.col('trans_date_trans_time') - F.lag(F.col('trans_date_trans_time'), 1).over(Windowspec))/3600))

full_df = full_df.fillna(-1)
full_df.show(3)

"""Checking for uniqness"""


def print_dupl_percent(df):
    for c in df.columns:
        print(c, df.select(F.countDistinct(F.col(c)) /
              full_df.count()).rdd.collect()[0])


print_dupl_percent(full_df)

# Transformation
custom_transformer_month = CustomTransformer(
    input_col="trans_month", output_col="trans_month")
full_df = custom_transformer_month.transform(full_df)

custom_transformer_day = CustomTransformer(
    input_col="trans_day", output_col="trans_day")
full_df = custom_transformer_day.transform(full_df)

custom_transformer_hour = CustomTransformer(
    input_col="trans_hour", output_col="trans_hour")
full_df = custom_transformer_hour.transform(full_df)

full_df = full_df.drop('index', 'trans_date_trans_time', 'trans_num', 'dob')


"""#EDA"""

transactions_df.createOrReplaceTempView('transactions')
cart_holder_df.createOrReplaceTempView('cart_holder')
merchant_df.createOrReplaceTempView('merchant')

job_trans = spark.sql("SELECT job,  sum(is_fraud) AS num_trans " +
                      "FROM cart_holder INNER JOIN transactions on cart_holder.cart_holder_id=transactions.cart_holder_id" +
                      " GROUP BY job ORDER BY num_trans DESC LIMIT 10")
job_pandas = job_trans.toPandas()
job_pandas.to_csv('output/q_job.csv')

"""### AMT"""

amt_pandas = spark.sql(
    "SELECT is_fraud, amt FROM transactions WHERE amt < 1000").toPandas()
amt_pandas.to_csv('output/q_amt.csv')

"""### Recency"""


def define_recency_segment(unix_time):
    hours = unix_time
    if hours < 0:
        return 'first'
    elif hours < 1:
        return 'recent'
    elif hours < 6:
        return 'within 6 hours'
    elif hours < 12:
        return 'after 6 hours'
    elif hours < 24:
        return 'after half-day'
    else:
        return 'after 24 hours'


recency_udf = F.udf(lambda x: define_recency_segment(x), StringType())
temp = full_df.withColumn('recency_segment', recency_udf(F.col('recency')))

recency_pandas_fraud = temp.filter(F.col("is_fraud") == 1).groupby(
    'recency_segment').count().toPandas()
recency_pandas_not_fraud = temp.filter(F.col("is_fraud") == 0).groupby(
    'recency_segment').count().toPandas()

recency_pandas_not_fraud.columns = ['recency_segment', "num_transactions"]
recency_pandas_not_fraud.to_csv('output/q_recency_not_fraud.csv')

recency_pandas_fraud.columns = ['recency_segment', "num_transactions"]
recency_pandas_fraud.to_csv('output/q_recency_fraud.csv')


"""## Feature selection"""

cat_cols = ['merchant', 'category', 'first', 'last',
            'gender', 'street', 'city', 'state', 'job']
time_cols = ['trans_hour_sin', 'trans_hour_cos', 'trans_month_sin',
             'trans_month_cos', 'trans_day_sin', 'trans_day_cos']
num_cols = ['amt', 'merch_lat', 'merch_long', 'cc_num',
            'zip', 'lat', 'long', 'city_pop', 'age', 'lat_dist', 'long_dist', 'recency', 'distance']
label = 'is_fraud'

"""## Selection of numerical *features*"""

assembler_num = VectorAssembler(
    inputCols=time_cols + num_cols, outputCol="features_num")

pipeline = Pipeline(stages=[assembler_num])
model = pipeline.fit(full_df)
# Fit the pipeline ==> This will call the transform functions for all transformers
data_numerical = model.transform(full_df)

selector = UnivariateFeatureSelector(featuresCol="features_num", outputCol="selectedFeatures_num",
                                     labelCol=label, selectionMode="fwe")
selector.setFeatureType("continuous").setLabelType(
    "categorical")  # .setSelectionThreshold(10)

result = selector.fit(data_numerical).transform(data_numerical)

indexers = [StringIndexer(
    inputCol='category', outputCol="category_indexed").setHandleInvalid("skip")]
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                          outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]
assembler = VectorAssembler(inputCols=['selectedFeatures_num']+[encoder.getOutputCol() for encoder in encoders],
                            outputCol="features")
pipeline = Pipeline(stages=indexers + encoders + [assembler])
model = pipeline.fit(result)
# Fit the pipeline ==> This will call the transform functions for all transformers
result = model.transform(result).select('is_fraud', 'features', 'amt')

train_data, test_data = stratified_split(
    result, label='is_fraud', test_size=0.3, seed=1)

"""# Models

#### Logistic Regression
"""

scaler = StandardScaler().setInputCol('features').setOutputCol('scaledFeatures')
scaler = scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

model = LogisticRegression(featuresCol='scaledFeatures', labelCol=label)
start = time.time()
model = model.fit(scaled_train_data)
end = time.time()

predictions = model.transform(scaled_test_data)
print("time: ", (end-start)/60)


evaluate(predictions, label)

"""Grid search of parameters"""


paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.02, 0.08])\
    .addGrid(model.elasticNetParam, [0.2, 0.6]).build()
area_pr_eval = BinaryClassificationEvaluator(
    rawPredictionCol='prediction', labelCol=label, metricName='areaUnderPR')
#MulticlassClassificationEvaluator(labelCol='label', metricLabel=1, metricName='fMeasureByLabel')

scv = StratifiedCrossValidator(
    estimator=LogisticRegression(featuresCol='scaledFeatures', labelCol=label),
    estimatorParamMaps=paramGrid,
    evaluator=area_pr_eval,
    numFolds=4
)

model_scv = scv.fit(scaled_train_data)

model_scv.getEstimatorParamMaps()[np.argmax(model_scv.avgMetrics)]

lr_metrics = evaluate(model_scv.transform(scaled_test_data), label)

"""#### Decision Tree"""

start = time.time()
dt = DecisionTreeClassifier(featuresCol='features',
                            labelCol='is_fraud', maxDepth=5, seed=1)
dtModel = dt.fit(train_data)
end = time.time()

dtPreds = dtModel.transform(test_data)
print("time: ", (end-start)/60)

evaluate(dtPreds, label)


"""Grid search of parameters"""

dt = DecisionTreeClassifier(featuresCol='features',
                            labelCol='is_fraud', maxDepth=2, seed=1)

paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [10, 20])
             .addGrid(dt.maxBins, [10, 20])
             # .addGrid(dt.minInstancesPerNode, [5, 10])
             .build())

# stratified
dtcv = StratifiedCrossValidator(estimator=dt,
                                estimatorParamMaps=paramGrid,
                                evaluator=area_pr_eval,
                                numFolds=4)
dtcvModel = dtcv.fit(train_data)

dtcvModel.getEstimatorParamMaps()[np.argmax(dtcvModel.avgMetrics)]

dtpreds = dtcvModel.transform(test_data)
dt_metrics = evaluate(dtpreds, label)

"""#### GBtree"""

gbt = GBTClassifier(maxIter=10, seed=1,
                    featuresCol='features', labelCol='is_fraud')
gbtModel = gbt.fit(train_data)
gbtPreds = gbtModel.transform(test_data)

evaluate(gbtPreds)

"""Grid search of parameters"""

gbt = GBTClassifier(maxIter=10, seed=1)

paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [3, 7])
             .addGrid(dt.stepSize, [0.06, 0.1])
             .addGrid(dt.maxIter, [10, 20])
             .build())

# stratified
gbtcv = CrossValidator(estimator=gbt,
                       estimatorParamMaps=paramGrid,
                       evaluator=area_pr_eval,
                       numFolds=4)
gbt_model = gbtcv.fit(train_data)

gbt_model.getEstimatorParamMaps()[np.argmax(gbt_model.avgMetrics)]

gbt_preds = gbt_model.transform(test_data)
evaluate(gbt_preds, label)

# transforming the feature importance in readable format


def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + \
            dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return (varlist.sort_values('score', ascending=False))


# obtainin top 20 important features
res = ExtractFeatureImp(gbtModel.featureImportances,
                        train_data, "features").head(20)


# creating and saving dataframe for further visualization
dic = {'name': [], 'score': []}
for _, i in res.iterrows():
    name, score = i[1], i[2]
    if "selectedFeatures_num" in name:
        name = name[21:]
    dic['name'].append(name)
    dic['score'].append(score)

df = pd.DataFrame(data=dic)
df.to_csv('output/feat_importance.csv')

"""### Final prediction"""

# set thresholds
thresholds = [0 + (i+1)/100 for i in range(9)] + \
    [0 + (i+1)/10 for i in range(10)]
metric = CostEvaluator(predictionCol='pred', labelCol='is_fraud')
metric_values = []

for thresh in tqdm(thresholds):
    predictions = gbtModel.transform(test_data)
    bin_udf = F.udf(lambda x: (float(x[1]) >= thresh) * 1, IntegerType())
    predictions = predictions.withColumn("pred", bin_udf(F.col("probability")))
    metric_values.append(metric.evaluate(predictions))

threshold_results = pd.DataFrame([thresholds, metric_values]).T
threshold_results.columns = ['threshold', 'loss']
threshold_results.to_csv("output/final_model_thresh.csv")


plt.plot(thresholds, metric_values)
print("Loss before model: ")
amt_val = test_data.filter("is_fraud = 1").select("amt").agg({'amt': 'sum'}).collect()[0][0]#.show()
print("Loss after model: ", threshold_results.loss.min())
print("Profit: ", amt_val - threshold_results.loss.min())
