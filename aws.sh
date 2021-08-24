#!/bin/bash

# Create S3 bucket
aws s3 mb s3://bucket-default-prediction-21

# Push pickled model to S3
aws s3 cp pickled_model.pickle s3://aws s3 mb s3://bucket-default-prediction-21

# ECR Repository (Here's how you create one:)
aws ecr create-repository --repository-name default-prediction-21

# SAM
sam init
sam build
sam deploy --guided