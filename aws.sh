#!/bin/bash

# Create S3 bucket
aws s3 mb s3://sam-sklearn-lam-132

# Push pickled model to S3
aws s3 cp pickled_model.pickle s3://sam-sklearn-lam-132

