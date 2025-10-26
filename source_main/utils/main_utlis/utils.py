import yaml 
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging
import sys
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Bank_Exception(e,sys)
    
    
def write_yaml_file(file_path:str,content:dict,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as yaml_file:
            yaml.dump(content,yaml_file)
    except Exception as e:
        raise Bank_Exception(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.array)->None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise Bank_Exception(e,sys)
    
    
def save_object(file_path:str,obj:object)->None:
    try:
        logging.info("Entered the save_object method of main_utils class")
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name,exist_ok=True)
            
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
        logging.info("Exited the save_object method of main_utils class")
    except Exception as e:
        raise Bank_Exception(e,sys)


def load_schema():
    with open("data_schema/schema.yaml", "r") as f:
        schema = yaml.safe_load(f)
    return schema["numeric_columns"], schema["categorical_columns"], schema["target_column"]

    
def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Bank_Exception(e,sys)
    
def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise Bank_Exception(e,sys)
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    report = {}
    try:
        logging.info("=== Model Evaluation Started ===")

        for model_name, model in models.items():
            logging.info(f"\n▶ Training model: {model_name}")

            param_grid = params.get(model_name, {})

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

            # Store results, including trained model
            report[model_name] = {
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "test_precision": round(test_precision, 4),
                "test_recall": round(test_recall, 4),
                "test_f1": round(test_f1, 4),
                "best_params": gs.best_params_,
                "best_model": best_model,  # 
            }

            logging.info(
                f"{model_name} Results → "
                f"Train Acc={train_accuracy:.3f}, Test Acc={test_accuracy:.3f}, "
                f"Precision={test_precision:.3f}, Recall={test_recall:.3f}, F1={test_f1:.3f}"
            )
            logging.info(f"Best Params for {model_name}: {gs.best_params_}")

        logging.info("=== Model Evaluation Completed ===")
        return report

    except Exception as e:
        logging.exception("Error occurred during model evaluation.")
        raise Bank_Exception(e, sys)