
import pandas as pd
import numpy as np
import os
import joblib
from io import StringIO
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from transformers import ColumnSelector
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

feature_columns_names = ['uuid','default', 'account_amount_added_12_24m', 'account_days_in_dc_12_24m',
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
model_dir = "models/"

if __name__ == '__main__':

    
    # read data
    train = pd.read_table('data/train.csv', sep = ";", index_col=0, names=feature_columns_names, dtype=feature_columns_dtype)

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
    
    joblib.dump(preprocessor, os.path.join(model_dir, "model.joblib"))

    print("saved model!")
