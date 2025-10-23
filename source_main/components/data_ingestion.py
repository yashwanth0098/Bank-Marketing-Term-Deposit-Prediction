import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mysql.connector
from dotenv import load_dotenv

from source_main.entity.artifact import DataIngestionArtifact
from source_main.entity.config import DataIngestionConfig
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging

load_dotenv() # load environment variables from .env file



MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

 
class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
                raise Bank_Exception(e, sys)
            
    def export_table_as_dataframe(self):
        try:
            database_name=self.data_ingestion_config.database_name
            table_name=self.data_ingestion_config.table_name
            connection=mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
            cursor=connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            rows=cursor.fetchall()
            column_names=[column[0] for column in cursor.description]
            df=pd.DataFrame(rows,columns=column_names)
            return df
        except Exception as e:
            raise Bank_Exception(e, sys)


    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_dir
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return feature_store_file_path
        except Exception as e:
            raise Bank_Exception(e, sys)
    
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_data,test_data=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("Performed train test split")
            
            logging.info("exited split_data_as_train_test method of Data Ingestion class")
            
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info("Created directory for train and test data")    
            
            train_data.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            
            logging.info("Exported train and test data")
            
            
        except Exception as e:
            raise Bank_Exception(e, sys)
        
        
    def initiate_data_ingestion(self):
        try:
            df = self.export_table_as_dataframe()
            self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)
            dataingestionartifacts = DataIngestionArtifact(
            train_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path
        )
            return dataingestionartifacts
        
        except Exception as e:
            raise Bank_Exception(e, sys)
