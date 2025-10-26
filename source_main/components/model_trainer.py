import os
import sys
import numpy as np
import pandas as pd

from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging
from source_main.entity.config import ModelTrainerConfig
from source_main.utils.ml_utlis.model.estimator import BankModel
from source_main.utils.main_utlis.utils import (
    save_object, load_object, load_numpy_array_data, evaluate_model
)
from source_main.utils.ml_utlis.metric.classification_metric import get_classification_metric
from source_main.entity.artifact import DataTransformationArtifact, ModelTrainerArtifact

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import mlflow
import dagshub

dagshub.init(repo_owner="yashwanth0098", repo_name="deposit_classification", mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise Bank_Exception(e, sys)

    def track_mlflow(self, model_name, model, classification_metric):
        """Track model performance metrics and parameters in MLflow."""
        try:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("Model", model_name)
                mlflow.log_metric("f1_score", classification_metric.f1_score)
                mlflow.log_metric("precision", classification_metric.precision_score)
                mlflow.log_metric("recall", classification_metric.recall_score)
                mlflow.log_metric("accuracy", classification_metric.accuracy_score)
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:  
            raise Bank_Exception(e, sys)

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train multiple models, evaluate, and select the best one."""
        try:
            # ----------------------------
            # Define models
            # ----------------------------
            models = {
                'log_regression': LogisticRegression(max_iter=500),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42),
                'adaboost': AdaBoostClassifier(random_state=42),
                'gradient_boost': GradientBoostingClassifier(random_state=42),
                'xgboost': XGBClassifier( eval_metric='logloss', random_state=42),
                'catboost': CatBoostClassifier(verbose=0, random_state=42)
            }

            # ----------------------------
            # Define hyperparameters
            # ----------------------------
            params = {
                'log_regression': {'penalty': ['l2'], 'solver': ['lbfgs']},
                'decision_tree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
                'random_forest': {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
                'adaboost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 1.0]},
                'gradient_boost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
                'xgboost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
                'catboost': {'iterations': [50, 100], 'depth': [4, 6], 'learning_rate': [0.01, 0.1]}
            }

            # ----------------------------
            # Evaluate all models
            # ----------------------------
            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # ----------------------------
            # Select best model based on F1 score
            # ----------------------------
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_f1'])
            best_model_info = model_report[best_model_name]
            best_model = best_model_info["best_model"] 
            best_model_score = best_model_info['test_f1']

            logging.info(f"Best model selected: {best_model_name} with F1 score: {best_model_score:.4f}")

            # ----------------------------
            # Train and log best model
            # ----------------------------
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_metric(y_true=y_train, y_pred=y_train_pred)
            self.track_mlflow(f"{best_model_name}_train", best_model, classification_train_metric)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(f"{best_model_name}_test", best_model, classification_test_metric)

            # ----------------------------
            # Save final model with preprocessor
            # ----------------------------
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            bank_model = BankModel(preprocessor=preprocessor, model=best_model)

            os.makedirs(os.path.dirname(self.model_trainer_config.trainer_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trainer_model_file_path, obj=bank_model)
            save_object("final_model.pkl", best_model)

            # ----------------------------
            # Create artifact
            # ----------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trainer_model_file_path,
                training_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model Trainer Artifact created successfully: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise Bank_Exception(e, sys)

    def initiate_model_trainer(self):
        """Load transformed arrays, train, and return model artifact."""
        try:
            train_array = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_trainer_artifact = self.train_model(X_train, X_test, y_train, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise Bank_Exception(e, sys)
