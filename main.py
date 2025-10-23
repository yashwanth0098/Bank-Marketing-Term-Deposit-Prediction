from source_main.components.data_ingestion import DataIngestion
from source_main.components.data_validation import DataValidation
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging
from source_main.entity.config import DataIngestionConfig, TrainingPipelineConfig, Data_ValidationConfig
import sys, os
from source_main.entity.artifact import DataIngestionArtifact   



if __name__=="__main__":
    try:
        TrainingPipelineConfig=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(TrainingPipelineConfig)
        dataingestion=DataIngestion(data_ingestion_config)
        logging.info("Exporting the data from the database")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info('Data Ingestion Completed')
        
        
        data_validation_config=Data_ValidationConfig(TrainingPipelineConfig)
        data_validation=DataValidation(data_validation_config,dataingestionartifact)
        logging.info("Starting data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("Data Validation Completed")
    except Exception as e:
        raise Bank_Exception(e, sys)
        