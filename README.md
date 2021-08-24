# Predicting probability of defauld

## Workflow

- Model training
- Model deployment
- Build, Test, and Deploy

## Technologies
- Docker + ECR: Container & Registy
- AWS Lambda: Serving API
- SAM: Serverless Framework 
- GitHub Actions: CI/CD

## Project Structure
```
|-- data
    |-- submission.csv
|--sam-app
    |-- service
         |-- app.py: source code lambda handler
         |-- Dockerfile: to build the Docker image
         |-- requirements.txt: dependencies
    |-- tests
         |-- unit
              |--test_handler.py: unit test/s for lambda handler
    |-- samconfig.toml: configured by SAM
    |-- template.yaml: A template that defines the application's AWS resources.
|-- src
    |-- models
        |-- baseline_model.py: Class for training the binary classification model
        |-- transformers.py: ColumnSelector transformer for preprocessing
|-- api_request_exmpl.json
|-- pickled_model.pickle
|-- Model.ipunb: model evaluation notebook
```

## Pre-requisites

* **python3.9**
* **Docker** 
* **awscli**
* **aws-sam-cli**

## Setup
AWS account with IAM user & required permissions
ECR Repository

## Steps

1. Model training (`/Model.ipynb`)
    1. Read and explore dataset (excluded from the current repo due to privacy reasons)
    2. Train baseline defualt prediction model
    3. Evaluate and сompare different models (Logistic Regression and Tree-based ensemble models) with a set of metrics
    4. Estimate feature inportance
    5. Generate predictions (`/data/submission.csv`)
    6. Dump the best estimator (`/pickled_model.pickle`)
  

