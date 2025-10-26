import os 
import pandas as pd
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFICATS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFICATS_DIR, exist_ok=True)
PICKLE_DIR = os.path.join(ARTIFICATS_DIR, "pickle_files")
os.makedirs(PICKLE_DIR, exist_ok=True)



""" Defining the common constants for the training pipeline """


TARGET_COLUMN = "y"
PIPELINE_NAME: str = "bank_deposit"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "bank_data.csv"


TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEME_FILE_PATH: str = os.path.join('data_schema', 'schema.yaml')

SAVE_MODEL_DIR= os.path.join('saved_models')
MODEL_FILE_NAME= "model.pkl"

""" Data Ingestion related constants """

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_DATABASE_NAME: str="bank_database"
DATA_INGESTION_TABLE_NAME: str = "bankdeposit"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

""" Data Validation related constants """
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

""" Data Transformation related constants """
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## Defining pickle file names for preprocessor, scaler and encoder

PREPROCESSOR_OBJ_FILE_NAME = os.path.join(PICKLE_DIR, "preprocessor.pkl")
SCALER_FILE_NAME = os.path.join(PICKLE_DIR, "scaler.pkl")
ENCODER_FILE_NAME = os.path.join(PICKLE_DIR, "encoder.pkl")


""" Model Trainer related constants """


MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME: str = "trainingbucket"
ARTIFACT_BUCKET_NAME: str = "artifactbucket"


