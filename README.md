# Machine Learning Case Study
## Predict probability of defauld

Workflow:

**Model training**
1. Read and explore given dataset (excluded from the current repo due to privacy reasons).
`/Model.ipynb` sections 1.1 - 1.3
2. Model training:
* Train baseline defualt prediction model.
* Evaluate and сompare different models (Logistic Regression and Tree-based ansamble models) with a set of metrics.
* Estimate feature inportance.
`/Model.ipynb` sections 1.4 - 1.6
3. Generate prediction, dump them to file, save model
`/Model.ipynb` sections 1.7 - 1.8
`/data/submission.csv`

**Model deployment**
Deploying Sklearn Machine Learning pipeline on AWS Lambda with SAM (Serverless Application Model)
1. Install and configure AWS CLI & create necessary users and roles.
2. Create S3 bucket and push the model
`/aws.sh`
3. Build a SAM Application
`/sam-app`
4. Configure AWS CloudFormation for DefaultPredictorFunction
`/sam-app/template.yml`
5. Design Lambda to output predictions
`/sam-app/predictor/app.py`
6. Build SAM app

**Expose the model with an API Endpoin**
`/api_request.json`
7. Test the application locally with
```
sam local start-api
curl -XPOST http://127.0.0.1:3000/classify -H 'Content-Type: application/json' -d api_request.json
```
8. Deploy the application to AWS and test the API
```
sam package --template-file template.yaml --s3-bucket sam-sklearn-lam-132 --output-template-file packaged.yaml
sam deploy --template-file packaged.yaml --stack-name SklearnLambdaStack --capabilities CAPABILITY_IAM
curl -XPOST https://${ServerlessRestApi}.execute-api.eu-central-1.amazonaws.com/Prod/predict/
```

