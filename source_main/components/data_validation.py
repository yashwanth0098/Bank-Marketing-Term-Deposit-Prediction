from source_main.entity.artifact import DataIngestionArtifact, DataValidationArtifact
from source_main.entity.config import Data_ValidationConfig
from source_main.logging.logger import logging
from source_main.exception.exception import Bank_Exception
import os, sys
from source_main.utils.main_utlis.utils import read_yaml_file, write_yaml_file

from source_main.constants.pipelineconstants import SCHEME_FILE_PATH
import pandas as pd
import numpy as np

class DataValidation:
    def __init__(self,data_validation_config:Data_ValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.schema_config=read_yaml_file(SCHEME_FILE_PATH)
        except Exception as e:
            raise Bank_Exception(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Bank_Exception(e,sys)
        
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame):
        try:
            number_of_columns=len(self.schema_config)
            
            logging.info(f"Expected number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {dataframe.columns}")
            if number_of_columns==len(dataframe.columns):
                return True
            return False
        except Exception as e:
            raise Bank_Exception(e,sys)
        
        
    def calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index (PSI) for a single feature."""
        try:
            # Remove NaN values
            expected = expected[~pd.isnull(expected)]
            actual = actual[~pd.isnull(actual)]

            # Create bins based on expected data
            quantiles = np.linspace(0, 1, buckets + 1)
            breakpoints = np.quantile(expected, quantiles)

            expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

            # Replace zeros to avoid division or log errors
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi_value = np.sum((expected_percents - actual_percents) *
                               np.log(expected_percents / actual_percents))
            return psi_value
        except Exception as e:
            raise Bank_Exception(e, sys)

    def detect_dataset_drift(self, base_df, current_df, psi_threshold=0.25) -> bool:
        """
        Detect dataset drift using Population Stability Index (PSI).
        PSI < 0.1: No significant drift
        0.1 ≤ PSI < 0.25: Moderate drift
        PSI ≥ 0.25: Significant drift
        """
        try:
            status = True
            report = {}

            for column in base_df.columns:
                if pd.api.types.is_numeric_dtype(base_df[column]):
                    psi_value = self.calculate_psi(base_df[column], current_df[column])
                    drift_status = psi_value >= psi_threshold
                    if drift_status:
                        status = False
                else:
                    psi_value = np.nan
                    drift_status = "Not Applicable (Non-numeric column)"

                report[column] = {
                    "psi_value": float(psi_value) if not np.isnan(psi_value) else None,
                    "drift_detected":  bool(drift_status) if isinstance(drift_status, (bool, np.bool_)) else drift_status
                }

            # Write PSI drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise Bank_Exception(e, sys)
        
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            
            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train dataframe does not have all columns"
                logging.error(error_message)
            status=self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test dataframe does not have all columns"
                logging.error(error_message)
                
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path),exist_ok=True)
                        
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            
            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise Bank_Exception(e,sys)