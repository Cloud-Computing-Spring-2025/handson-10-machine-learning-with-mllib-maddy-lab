import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("CustomerChurnPrediction").getOrCreate()

# Load dataset
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)

# Task 1: Preprocessing
def preprocess_data(df):
    df = df.na.fill({'TotalCharges': 0})
    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    
    indexers = [StringIndexer(inputCol=col, outputCol=col+"Index").fit(df) for col in categorical_cols + ['Churn']]
    for indexer in indexers:
        df = indexer.transform(df)
    
    encoder = OneHotEncoder(inputCols=[col+"Index" for col in categorical_cols],
                            outputCols=[col+"Vec" for col in categorical_cols])
    df = encoder.fit(df).transform(df)

    assembler = VectorAssembler(inputCols=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] + [col+"Vec" for col in categorical_cols],
                                outputCol="features")
    final_df = assembler.transform(df).select('features', col('ChurnIndex').alias('label'))

    # Save preview CSV
    os.makedirs("output/task1_preprocessed", exist_ok=True)
    final_df.limit(5).toPandas().to_csv("output/task1_preprocessed/preview.csv", index=False)

    return final_df

# Task 2: Logistic Regression
def train_logistic_regression(df):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol='features', labelCol='label')
    model = lr.fit(train)
    predictions = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)

    # Save AUC result
    os.makedirs("output/task2_logistic_auc", exist_ok=True)
    pd.DataFrame([{"Model": "Logistic Regression", "AUC": round(auc, 2)}]).to_csv("output/task2_logistic_auc/auc.csv", index=False)

# Task 3: Feature Selection
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    selected_df = selector.fit(df).transform(df).select("selectedFeatures", "label")

    os.makedirs("output/task3_selected_features", exist_ok=True)
    selected_df.limit(5).toPandas().to_csv("output/task3_selected_features/preview.csv", index=False)

# Task 4: Hyperparameter Tuning and Comparison
def hyperparameter_tuning(df):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    models = [
        ('LogisticRegression', LogisticRegression(featuresCol='features', labelCol='label'),
         ParamGridBuilder().addGrid(LogisticRegression.regParam, [0.01, 0.1]).addGrid(LogisticRegression.maxIter, [10, 20]).build()),

        ('DecisionTree', DecisionTreeClassifier(featuresCol='features', labelCol='label'),
         ParamGridBuilder().addGrid(DecisionTreeClassifier.maxDepth, [5, 10]).build()),

        ('RandomForest', RandomForestClassifier(featuresCol='features', labelCol='label'),
         ParamGridBuilder().addGrid(RandomForestClassifier.numTrees, [20, 50]).addGrid(RandomForestClassifier.maxDepth, [10, 15]).build()),

        ('GBT', GBTClassifier(featuresCol='features', labelCol='label'),
         ParamGridBuilder().addGrid(GBTClassifier.maxDepth, [5, 10]).addGrid(GBTClassifier.maxIter, [10, 20]).build())
    ]

    results = []
    for name, model, paramGrid in models:
        print(f"Tuning {name}...")
        cv = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cvModel = cv.fit(train)
        bestModel = cvModel.bestModel
        predictions = bestModel.transform(test)
        auc = evaluator.evaluate(predictions)
        results.append({
            "Model": name,
            "AUC": round(auc, 2),
            "BestParams": str(bestModel.extractParamMap())
        })

    os.makedirs("output/task4_model_comparison", exist_ok=True)
    pd.DataFrame(results).to_csv("output/task4_model_comparison/model_scores.csv", index=False)

# Execute all tasks
final_df = preprocess_data(df)
train_logistic_regression(final_df)
feature_selection(final_df)
hyperparameter_tuning(final_df)

spark.stop()
