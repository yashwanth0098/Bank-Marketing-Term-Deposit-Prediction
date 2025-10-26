from datetime import datetime
import os 
from source_main.constants import pipelineconstants
print(pipelineconstants.PIPELINE_NAME)
print(pipelineconstants.ARTIFACT_DIR)


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=pipelineconstants.PIPELINE_NAME
        self.artifact_name=pipelineconstants.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join('final_model')
        self.timestamp: str=timestamp
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,pipelineconstants.DATA_INGESTION_DIR_NAME)
        
        self.feature_store_dir:str=os.path.join(
            self.data_ingestion_dir,pipelineconstants.DATA_INGESTION_FEATURE_STORE_DIR,pipelineconstants.FILE_NAME)

        self.training_file_path:str = os.path.join(
            self.data_ingestion_dir, pipelineconstants.DATA_INGESTION_FEATURE_STORE_DIR, pipelineconstants.TRAIN_FILE_NAME
        )
        self.testing_file_path:str = os.path.join(
            self.data_ingestion_dir,pipelineconstants.DATA_INGESTION_FEATURE_STORE_DIR, pipelineconstants.TEST_FILE_NAME
        )
        self.train_test_split_ratio:float=pipelineconstants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.database_name:str=pipelineconstants.DATA_INGESTION_DATABASE_NAME
        self.table_name:str=pipelineconstants.DATA_INGESTION_TABLE_NAME
        
        
class Data_ValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        self.data_validation_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,pipelineconstants.DATA_VALIDATION_DIR_NAME
            )
        self.valid_dir:str=os.path.join(
            self.data_validation_dir,pipelineconstants.DATA_VALIDATION_VALID_DIR
            )
        self.invalid_dir:str=os.path.join(
            self.data_validation_dir,pipelineconstants.DATA_VALIDATION_INVALID_DIR
            )
        self.valid_train_file_path:str=os.path.join(
            self.valid_dir,pipelineconstants.TRAIN_FILE_NAME          
            )
        self.valid_test_file_path:str=os.path.join(
            self.valid_dir,pipelineconstants.TEST_FILE_NAME
            )
        self.invalid_train_file_path:str=os.path.join(
            self.invalid_dir,pipelineconstants.TRAIN_FILE_NAME
            )
        self.invalid_test_file_path:str=os.path.join(
            self.invalid_dir,pipelineconstants.TEST_FILE_NAME
            )
        self.drift_report_file_path:str=os.path.join(
            self.data_validation_dir,pipelineconstants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            pipelineconstants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
            )
        
class DataTransformationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,
            pipelineconstants.DATA_TRANSFORMATION_DIR_NAME
            )
        self.transformed_data_dir:str=os.path.join(
            self.data_transformation_dir,
            pipelineconstants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
            )
        self.transformed_train_file_path = os.path.join(
            self.transformed_data_dir,
            pipelineconstants.TRAIN_FILE_NAME.replace('.csv', '.npy')
        )
        self.transformed_test_file_path = os.path.join(
            self.transformed_data_dir,
            pipelineconstants.TEST_FILE_NAME.replace('.csv', '.npy')
        )
       
        self.transformed_object_file_path = os.path.join(
        self.data_transformation_dir,
        pipelineconstants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, "preprocessor.pkl")
        
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,pipelineconstants.MODEL_TRAINER_DIR_NAME
            )
        self.trainer_model_file_path:str=os.path.join(
            self.model_trainer_dir,pipelineconstants.MODEL_TRAINER_TRAINED_MODEL_DIR,
            pipelineconstants.MODEL_FILE_NAME
        )
        self.excpected_accuracy:float=pipelineconstants.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold:float=pipelineconstants.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD