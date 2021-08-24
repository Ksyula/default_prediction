#!/bin/bash

# Create S3 bucket
aws s3 mb s3://bucket-default-prediction-21

# Push pickled model to S3
aws s3 cp pickled_model.pickle s3://aws s3 mb s3://bucket-default-prediction-21

# ECR Repository (Here's how you create one:)
aws ecr create-repository --repository-name default-prediction-21

#{
#    "repository": {
#        "repositoryArn": "arn:aws:ecr:eu-central-1:918203234730:repository/default-prediction-21",
#        "registryId": "918203234730",
#        "repositoryName": "default-prediction-21",
#        "repositoryUri": "918203234730.dkr.ecr.eu-central-1.amazonaws.com/default-prediction-21",
#        "createdAt": "2021-08-24T08:24:50+02:00",
#        "imageTagMutability": "MUTABLE",
#        "imageScanningConfiguration": {
#            "scanOnPush": false
#        },
#        "encryptionConfiguration": {
#            "encryptionType": "AES256"
#        }
#    }
#}