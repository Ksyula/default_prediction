
import pandas as pd
import numpy as np
import os
import joblib
from io import StringIO
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

feature_columns_names = ['uuid', 'default', 'account_amount_added_12_24m', 'account_days_in_dc_12_24m',
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
                       'sum_paid_inv_0_12m', 'time_hours', 'worst_status_active_inv']
                       
feature_columns_dtypes = {'uuid': "object", 'default': "float64", 'account_amount_added_12_24m' : "int64", 'account_days_in_dc_12_24m' : "float64", 'account_days_in_rem_12_24m' : "float64", 
                        'account_days_in_term_12_24m' : "float64", 'account_incoming_debt_vs_paid_0_24m' : "float64", 'account_status' : "float64", 
                        'account_worst_status_0_3m' : "float64", 'account_worst_status_12_24m' : "float64", 'account_worst_status_3_6m' : "float64", 
                        'account_worst_status_6_12m' : "float64", 'age' : "int64", 'avg_payment_span_0_12m' : "float64", 'avg_payment_span_0_3m' : "float64", 
                        'merchant_category' : "object", 'merchant_group' : "object", 'has_paid' : "bool", 'max_paid_inv_0_12m' : "float64", 
                        'max_paid_inv_0_24m' : "float64", 'name_in_email' : "object", 'num_active_div_by_paid_inv_0_12m' : "float64", 'num_active_inv' : "int64", 
                        'num_arch_dc_0_12m' : "int64", 'num_arch_dc_12_24m' : "int64", 'num_arch_ok_0_12m' : "int64", 'num_arch_ok_12_24m' : "int64", 
                        'num_arch_rem_0_12m' : "int64", 'num_arch_written_off_0_12m' : "float64", 'num_arch_written_off_12_24m' : "float64", 'num_unpaid_bills' : "int64", 
                        'status_last_archived_0_24m' : "int64", 'status_2nd_last_archived_0_24m' : "int64", 'status_3rd_last_archived_0_24m' : "int64", 
                        'status_max_archived_0_6_months' : "int64", 'status_max_archived_0_12_months' : "int64", 'status_max_archived_0_24_months' : "int64", 
                        'recovery_debt' : "int64", 'sum_capital_paid_account_0_12m' : "int64", 'sum_capital_paid_account_12_24m' : "int64", 'sum_paid_inv_0_12m' : "int64", 
                        'time_hours' : "float64", 'worst_status_active_inv' : "float64"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [pd.read_csv(
                file, 
                header=None, 
                sep = ";",
                names=feature_columns_names, dtype=feature_columns_dtypes) for file in input_files]

    train = pd.concat(raw_data)
    train = train.set_index("uuid")

    # define features
    categorical_low_card = [col for col in train.columns if col.find("status") != -1]
    categorical_high_card = ["merchant_category", "merchant_group", "name_in_email"]
    binary = ["has_paid"]
    numerical = list(set(train.columns) - set(categorical_low_card + categorical_high_card + binary) - set(['default']))

    # Preprocessing pipeline
    # Numeric features transforming Pipeline
    num_transformer = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ])
    # Categorical features transforming Pipeline
    cat_low_card_transformer = Pipeline(steps=[
        ('cat_low_imputer', SimpleImputer(strategy="most_frequent"))
    ])

    cat_high_card_transformer = Pipeline(steps=[
        ('cat_high_imputer', OrdinalEncoder())
    ])

    # Binary features transforming Pipeline
    binary_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical),
            ('cat_low_card', cat_low_card_transformer, categorical_low_card),
            ('cat_high_card', cat_high_card_transformer, categorical_high_card),
            ('binary', binary_transformer, binary)
        ])
    
    preprocessor.fit(train)
    
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")

    
def input_fn(input_data, content_type):
    """Parse input data
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        
        
def predict_fn(input_data, model):
    """Preprocess input data

    The model is a preprocessor = use .transform().
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
        
        