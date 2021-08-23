
import boto3
import os
import json
import pickle
import pandas as pd
s3 = boto3.client('s3')
s3_bucket = os.environ['s3_bucket']
model_name = os.environ['model_name']
temp_file_path = '/tmp/' + model_name
from sklearn.ensemble import RandomForestClassifier


def lambda_handler(event, context):
    # Parse input
    body = event['body']
    with open(body) as f:
        input = json.load(f)['data']
    input = pd.DataFrame([tuple(input["data"])], columns=['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
                                                       'account_days_in_rem_12_24m', 'account_days_in_term_12_24m',
                                                       'account_incoming_debt_vs_paid_0_24m', 'account_status',
                                                       'account_worst_status_0_3m', 'account_worst_status_12_24m',
                                                       'account_worst_status_3_6m', 'account_worst_status_6_12m', 'age',
                                                       'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'merchant_category',
                                                       'merchant_group', 'has_paid', 'max_paid_inv_0_12m',
                                                       'max_paid_inv_0_24m', 'name_in_email',
                                                       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
                                                       'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
                                                       'num_arch_ok_12_24m', 'num_arch_rem_0_12m',
                                                       'num_arch_written_off_0_12m', 'num_arch_written_off_12_24m',
                                                       'num_unpaid_bills', 'status_last_archived_0_24m',
                                                       'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
                                                       'status_max_archived_0_6_months', 'status_max_archived_0_12_months',
                                                       'status_max_archived_0_24_months', 'recovery_debt',
                                                       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
                                                       'sum_paid_inv_0_12m', 'time_hours', 'worst_status_active_inv'])


    # Download pickled model from S3 and unpickle
    s3.download_file(s3_bucket, model_name, temp_file_path)
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)
    # Predict probability of default
    prediction = model.predict_proba(input)[:, 1]
    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": list(prediction),
        }),
    }