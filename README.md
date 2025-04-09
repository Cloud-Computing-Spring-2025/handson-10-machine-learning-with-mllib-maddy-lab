
# Customer Churn Prediction with MLlib

This project uses **Apache Spark MLlib** to predict customer churn based on structured customer data. You'll learn how to preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---

## 🔍 Dataset Overview

The dataset used is `customer_churn.csv`, which includes the following columns:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn`, etc.

---

## ⚙️ Setup Instructions

### Prerequisites

- Apache Spark installed
- Python environment with `pyspark` and `pandas` installed
- Place `customer_churn.csv` inside the root directory of the project

### Run the code

```bash
spark-submit customer-churn-analysis.py
```

---

## 🧠 Task 1: Data Preprocessing and Feature Engineering

### Objective

Prepare data for training by encoding categorical variables, assembling features, and handling missing values.

### Steps

1. **Missing Values**:
   ```python
   df = df.na.fill({'TotalCharges': 0})
   ```

2. **Encoding Categorical Columns**:
   ```python
   indexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
   df = indexer.fit(df).transform(df)
   ```

3. **One-Hot Encoding**:
   ```python
   encoder = OneHotEncoder(inputCols=["genderIndex", "PhoneServiceIndex", "InternetServiceIndex"],
                           outputCols=["genderVec", "PhoneServiceVec", "InternetServiceVec"])
   df = encoder.fit(df).transform(df)
   ```

4. **Feature Vector Assembly**:
   ```python
   assembler = VectorAssembler(
       inputCols=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'genderVec', 'PhoneServiceVec', 'InternetServiceVec'],
       outputCol="features")
   final_df = assembler.transform(df).select('features', col('ChurnIndex').alias('label'))
   ```

5. **Save Output**:
   ```python
   preview_df = final_df.limit(5).toPandas()
   preview_df.to_csv("output/task1_preprocessed/preview.csv", index=False)
   ```

### ✅ Sample Output:
```
+--------------------+-----------+
|features            |label      |
+--------------------+-----------+
|[0.0,12.0,29.85,... |0.0        |
...
```

---

## 🔎 Task 2: Train and Evaluate Logistic Regression Model

### Objective

Train a logistic regression model and evaluate it using Area Under the ROC Curve (AUC).

### Steps

1. **Train/Test Split**:
   ```python
   train, test = df.randomSplit([0.8, 0.2], seed=42)
   ```

2. **Train the Model**:
   ```python
   lr = LogisticRegression(featuresCol='features', labelCol='label')
   model = lr.fit(train)
   ```

3. **Evaluate Using AUC**:
   ```python
   evaluator = BinaryClassificationEvaluator(labelCol="label")
   auc = evaluator.evaluate(model.transform(test))
   ```

4. **Save Results**:
   ```python
   pd.DataFrame([{"Logistic Regression AUC": auc}]).to_csv("output/task2_logistic_regression/auc_result.csv", index=False)
   ```

### ✅ Sample Output:
```text
Logistic Regression AUC: 0.83
```

---

## 📊 Task 3: Feature Selection using Chi-Square Test

### Objective

Select the top 5 most relevant features using Chi-Square selection.

### Steps

1. **Apply ChiSqSelector**:
   ```python
   selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
   selected_df = selector.fit(df).transform(df).select("selectedFeatures", "label")
   ```

2. **Export Result**:
   ```python
   selected_df.limit(5).toPandas().to_csv("output/task3_feature_selection/top5_features.csv", index=False)
   ```

### ✅ Sample Output:
```
+--------------------+-----------+
|selectedFeatures    |label      |
+--------------------+-----------+
|[0.0,29.85,0.0,...  |0.0        |
...
```

---

## 🧪 Task 4: Hyperparameter Tuning & Model Comparison

### Objective

Compare four models using 5-fold Cross-Validation with parameter tuning.

### Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted Tree (GBT)

### Steps

1. **Setup Evaluator**:
   ```python
   evaluator = BinaryClassificationEvaluator(labelCol="label")
   ```

2. **Define and Tune Models**:
   ```python
   cv = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
   ```

3. **Evaluate and Log Best Results**:
   ```python
   pd.DataFrame(results).to_csv("output/task4_model_comparison/model_comparison.csv", index=False)
   ```

### ✅ Sample Output:
```
Model,Best AUC,Best Params
LogisticRegression,0.84,regParam=0.01, maxIter=20
DecisionTree,0.77,maxDepth=10
RandomForest,0.86,numTrees=50, maxDepth=15
GBT,0.88,maxDepth=10, maxIter=20
```

---

## 📁 Folder Structure

```bash
.
├── customer_churn.csv
├── customer-churn-analysis.py
└── output/
    ├── task1_preprocessed/preview.csv
    ├── task2_logistic_regression/auc_result.csv
    ├── task3_feature_selection/top5_features.csv
    └── task4_model_comparison/model_comparison.csv
```
