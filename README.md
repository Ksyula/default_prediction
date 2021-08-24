# Predicting probability of defauld

## Workflow

- Model training
- Generate predictions
- Build, Test, and Deploy to AWS

## Technologies
- Docker + ECR: Container & Registy
- AWS Lambda: Serving API
- SAM: Serverless Framework 
- GitHub Actions: CI/CD

## Project Structure
```
|-- data
    |-- submission.csv
|-- service
     |-- app.py: source code lambda handler
     |-- Dockerfile: to build the Docker image
     |-- requirements.txt: dependencies
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
- AWS account with IAM user & required permissions
- ECR Repository

## Steps

### Model training (`/Model.ipynb`)
1. Read and explore dataset (excluded from the current repo due to privacy reasons)
2. Train baseline defualt prediction model
3. Evaluate and сompare different models (Logistic Regression and Tree-based ensemble models) with a set of metrics
4. Estimate feature inportance
5. Generate predictions (`/data/submission.csv`)
6. Dump the best estimator (`/pickled_model.pickle`)

### Build
1. Init sam to generate project structure
2. Build sam 
3. Deploy image to AWS ECR
4. AWS will integrate image with Lambda function


