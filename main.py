from source_main.components.data_ingestion import DataIngestion
from source_main.components.data_validation import DataValidation
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging
from source_main.entity.config import DataIngestionConfig, TrainingPipelineConfig, Data_ValidationConfig, DataTransformationConfig,ModelTrainerConfig
import sys, os
from source_main.entity.artifact import DataIngestionArtifact   
from source_main.components.data_transformation import DataTransformation
from source_main.components.model_trainer import ModelTrainer


if __name__=="__main__":
    try:
        trainingpipelieconfig=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(trainingpipelieconfig)
        dataingestion=DataIngestion(data_ingestion_config)
        logging.info("Exporting the data from the database")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info('Data Ingestion Completed')
        
        
        data_validation_config=Data_ValidationConfig(trainingpipelieconfig)
        data_validation=DataValidation(data_validation_config,dataingestionartifact)
        logging.info("Starting data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("Data Validation Completed")
        
        data_transformation_config=DataTransformationConfig(trainingpipelieconfig)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Starting data transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation Completed")
        
    
        
        logging.info("Model Trainig started")
        model_trainer_config=ModelTrainerConfig(trainingpipelieconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer=model_trainer.initiate_model_trainer()
        
        logging.info("Model Trainer  artifacts created ")
    except Exception as e:
        raise Bank_Exception(e, sys)
    
        