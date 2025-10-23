import os 
import pandas as pd
import numpy as np


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