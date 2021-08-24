# Model training and validation
## Observations during exploratory data analysis 
1. Missing rate: 
* 40% of features in the dataset have missing values; 
* 8 features have almost 50% of missing values.
To deal with this I will use Imputer transformer later during data processing.

2. The provided dataset.csv was splitted into train/test by `default` 
target column values presence.

3. Dataset is drastically imbalanced - there is skewness in target values. 
Minority class is ~1,5% of the training dataset.

4. There are no significant correlations between features and the target value (the heights corr value is 0.2 for avg_payment_span_0_12m);
There is multicollinearity - some features are positively correlated with each other, which can be observed on the heatmaps in `Model.ipynb`.
To deal with multicollinearity I will exclude correlated features while passing featureset to linear models.

## Feature engineering steps
1. Check the feature types and data format.
2. Extract categorical and numerical features as they require different processing transformations.
I also split categorical features into those which have low cardinality values and high cardinality values. 
Having not many data points it is a bad practice to explode dataset via OneHotEncoding of high cardinality categorical features,
so that I will apply Target encoding to such features during data processing.
3. Check for outliers. Almost all numerical features have on average 5%-10% of outliers detected 
by the interquartile range approach (Boxplot). Linear models are especially sensitive to outliers. 
As a future improvement, an outlier removal could be incorporating into the data processing pipeline.

## Building a model
1. Define, train and evaluate various models. The model base is implemented in `BaselineModel` class.
As a baseline model, I chose a simple logistic regression. I'll compare it with 3 other types of models:
- LogisticRegression without multicollinearity
- RandomForestClassifier
- LGBMClassifier
2. For model validation and hyperparameter tuning I use `GridSearchCV` with `StratifiedKFold` framework.
For every set of hyperparameters, training is running on a part of the training dataset and evaluation procedure on a hold-out set.
This prevents overfitting, searches for the best hyperparameters, and ensure unbiased validation.
3. Training pipeline is build in `BaselineModel` and contains two steps:
- preprocessing input data (`StandardScaler, OrdinalEncoder, TargetEncoder, ColumnSelector, SimpleImputer`)
- training classifier
`GridSearchCV` is fitted by a training pipeline to prevent data leakage.

## Model evaluation & comparison
1. Since we have imbalanced target variable, the metrics for evaluation should be selected accordingly. 
For the evaluation and comparison I will use following metrics:
- F1 score
- Precision recall curve
- Avg precision score - is a numeric representation of precision recall curve
- ROC AUC score
You can find detailed model comparison in `Model.ipynb`

## Feature importance
Look at impurity-based feature importances prodiced by RandomForest estimator in `Model.ipynb`.
1. `age` and `avg_payment_span_0_12m` are significantly important features.
2. There are Top-5 features of the highest importance and the importance of the rest features is slightly decreasing. 
3. `merchant_category` is categorical feature with high cardinality encoded with Target encoder plays a great role.

## Future work
1. Train and evaluate Deep Learning classifiers if more datapoints could be provided.
2. Test and evaluate different transformers in the preprocessing stage especially for categorical variables.
3. Implement outliers removal in the preprocessing pipeline.



