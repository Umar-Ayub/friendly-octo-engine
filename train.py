# Standard Library Imports
import json
import logging
import os

# General Imports
import pandas as pd
import warnings

# SKLearn Imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay

import mlflow
import mlflow.sklearn
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
PWD = os.getcwd()
config_file = PWD + '/config.json'

class PreprocessData:
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def split_data(self):
        df_X = self.df.drop("y", axis=1)
        df_label = self.df["y"]
        # Make LogReg Pipeline
        RANDOM_STATE=self.config["data"]["RANDOM_STATE"]

        X_train, X_test, y_train, y_test = train_test_split(
            df_X,
            df_label,
            random_state=RANDOM_STATE
            )
        return X_train, X_test, y_train, y_test

class CreatePipeline:
    def __init__(self, config):
        self.config = config
    
    def create_classifier(self):
        numeric_features = self.config["data"]["NUMERIC_FEATURES"]
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_features = self.config["data"]["CATEGORICAL_FEATURES"]
        categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = Pipeline(
            steps=[("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=config["model"]["MAX_ITER"]))]
        )
        return clf



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
    except Exception as e:
        logger.exception(
            "Unable to load config file. Error: %s", e
        )
    
    try:
        df = pd.read_csv(config["data"]["DATA_LOCATION"])
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split Data
    obj = PreprocessData(df, config)
    X_train, X_test, y_train, y_test = obj.split_data()

    # Make LogReg Pipeline
    clf = CreatePipeline(config).create_classifier()

    with mlflow.start_run():
        mlflow.log_param('max_iter',config["model"]["MAX_ITER"])

        clf.fit(X_train, y_train)
        print("model score: %.3f" % clf.score(X_test, y_test))
        mlflow.log_metric("model_score",clf.score(X_test, y_test))

        tprobs = clf.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, clf.predict(X_test)))
        mlflow.log_param("classification_report",
                           classification_report(y_test, clf.predict(X_test)))

        print('Confusion matrix:')
        print(confusion_matrix(y_test, clf.predict(X_test)))
        mlflow.log_param("confusion_matrix",
                           confusion_matrix(y_test, clf.predict(X_test)))
        print(f'AUC: {roc_auc_score(y_test, tprobs)}')
        mlflow.log_metric("AUC",roc_auc_score(y_test, tprobs))
        RocCurveDisplay.from_estimator(estimator=clf,X= X_test, y=y_test)

        # save the model to disk and mlflow
        filename = config["model"]["SAVE_LOCATION"]
        pickle.dump(clf, open(filename, 'wb'))
        mlflow.sklearn.log_model(clf, "model")

