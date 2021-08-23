
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from src.models.transformers import ColumnSelector
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import sklearn.metrics as sk_metrics
from category_encoders.target_encoder import TargetEncoder
from typing import Dict, Any
import pandas as pd
import logging


class BaselineModel:

    """Class for training the binary classification model

    Parameters
    ----------
    model_config: Dict
                  Config contains features for training, 
                  classifier along with its parameters for gridsearch
    
    The Baseline model has pipeline definition with the main transformers and classifier.

    train_model function does the split of train and test dataset, 
    cv and evaluation of model on test set.

    """
    def __init__(self, model_config):
        self.model_name = model_config['model_name']
        self.categorical_high_card = model_config['categorical_high_card']
        self.categorical_low_card = model_config['categorical_low_card']
        self.num_features = model_config['numerical_features']
        self.binary_features = model_config['binary_features']
        self.target_col = model_config['target_column']
        self.clf_params = model_config['classifier_params']
        self.clf = model_config['classifier']

    @property
    def make_base_pipeline(self) -> Pipeline:
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
            ('cat_high_imputer', TargetEncoder(handle_unknown='ignore'))
        ])

        # Binary features transforming Pipeline
        binary_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat_low_card', cat_low_card_transformer, self.categorical_low_card),
                ('cat_high_card', cat_high_card_transformer, self.categorical_high_card),
                ('binary', binary_transformer, self.binary_features)
            ])

        model_pipe = Pipeline(
            [
                (
                    "col_selector",
                    ColumnSelector(
                        columns=self.num_features + self.categorical_low_card + self.categorical_high_card + self.binary_features
                    )
                ),
                ("proc", preprocessor),
                ("clf", self.clf),
            ]
        )

        return model_pipe


    def train_pipeline(self, X_train, y_train) -> Pipeline:
        """takes pipe from model config for training
        returns best estimator
        """

        skf = StratifiedKFold(n_splits=3)

        clf = GridSearchCV(
            self.make_base_pipeline,
            param_grid=self.clf_params,
            cv=list(skf.split(X_train, y_train)),
            n_jobs=1,
            verbose=3,
            refit=True
        )

        clf.fit(X_train, y_train)

        return clf.best_estimator_


    def compute_metrics(self, pipe, X_test, y_test) -> Dict[str, Any]:
        """
        Given the trained pipeline and test set, 
        compute binary classification metrics

        """
    
        predictions = pipe.predict(X_test)
        pred_proba = pipe.predict_proba(X_test)[:, 1]

        model_metrics = {}

        model_metrics["f1_score"] = sk_metrics.f1_score(
            y_test, predictions
        )
        
        model_metrics["avg_precision_score"] = sk_metrics.average_precision_score(
            y_test, pred_proba
        )

        precision, recall, pr_thresholds = sk_metrics.precision_recall_curve(y_test, pred_proba)
        model_metrics["roc_auc_score"] = sk_metrics.roc_auc_score(y_test, pred_proba)
        model_metrics["precision"] = precision
        model_metrics["recall"] = recall
        model_metrics["pr_thresholds"] = pr_thresholds
        model_metrics["pr_curve_auc"] = round(sk_metrics.auc(model_metrics['recall'], model_metrics['precision']), 3)

        logging.info(f'self.model_name: roc_auc_score is: {model_metrics["roc_auc_score"]}')
        
        
        return model_metrics

    def make_train_test_split(self, dataset: pd.DataFrame):
        """
        Make split of dataset into train and test

        """
    
        y = dataset[self.target_col]
        X = dataset.drop(self.target_col, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=17
        )

        return X_train, X_test, y_train, y_test


    def train_model(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Given the dataset, train model and compute metrics
        """
        
        X_train, X_test, y_train, y_test = self.make_train_test_split(dataset)
        
        pipe = self.train_pipeline(X_train, y_train)
        eval_metrics = self.compute_metrics(pipe, X_test, y_test)

        model_dict = {
            "model_name": self.model_name,
            "model": pipe,
            "model_metrics": eval_metrics
        }


        return model_dict
