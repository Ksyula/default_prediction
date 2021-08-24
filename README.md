# Predicting probability of defaul

## Workflow

- Model training
- Generate predictions
- Deploy the model to AWS using SageMaker

## Project Structure
```
|-- data
    |-- submission.csv
|-- sagemaker
     |-- data
     |-- models
     |-- AWS - SageMaker.ipynb
     |-- featurizer_local.py
     |-- featurizer_remote.py
|-- src
    |-- models
        |-- baseline_model.py: Class for training the binary classification model
        |-- transformers.py: ColumnSelector transformer for preprocessing
|-- api_request_exmpl.json
|-- pickled_model.pickle
|-- Model.ipunb: model evaluation notebook
```

## Pre-requisites

* **python3.8**
* **aws-cli**

## Steps

### Model training (`/Model.ipynb`)
1. Read and explore dataset (excluded from the current repo due to privacy reasons)
2. Train baseline defualt prediction model
3. Evaluate and сompare different models (Logistic Regression and Tree-based ensemble models) with a set of metrics
4. Estimate feature inportance
5. Generate predictions (`/data/submission.csv`)
6. Dump the best estimator (`/pickled_model.pickle`)

### Model deployment (`sagemaker/AWS - SageMaker.ipynb`)
1. Define Sagemaker session and role
2. Preprocessing data and train the model
3. Create SageMaker Scikit Estimator
4. Batch transform training data
5. Fit a Tree-based Model with the preprocessed data
6. Serial Inference Pipeline with Scikit preprocessor and classifier
7. Deploy model
8. Make a request to the pipeline endpoint


