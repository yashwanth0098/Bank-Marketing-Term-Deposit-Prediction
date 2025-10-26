import sys , os
import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline

from source_main.constants.pipelineconstants import TARGET_COLUMN

from source_main.entity.artifact import  DataTransformationArtifact, DataValidationArtifact
from source_main.entity.config import DataTransformationConfig   
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging
from source_main.utils.main_utlis.utils import save_object, save_numpy_array_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from source_main.utils.main_utlis.utils import load_schema




class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig): 
        try:
            self.data_validation_artifact : DataValidationArtifact = data_validation_artifact
            self.data_transformation_config : DataTransformationConfig  = data_transformation_config
        except Exception as e:
            raise Bank_Exception(e,sys)
        
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Bank_Exception(e,sys) 
        
    def get_data_transformer_object(self)->ColumnTransformer:
        
        logging.info("Entered the get_data_transformer_object method of Data_Transformation class")
        try:
            numeric_columns, categorical_columns, target_column = load_schema()

            # Define transformations for numeric columns
            numeric_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Define transformations for categorical columns
            categorical_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_pipeline, numeric_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise Exception(e,sys)
        
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Entered the initiate_data_transformation method of Data_Transformation class")
            
            #reading training and testing file
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            print("Train columns:", train_df.columns.tolist())
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            print("Test columns:", test_df.columns.tolist())
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessor object")
            

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].map({'yes':1,'no':0})
        

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].map({'yes':1,'no':0})
            
            preprocessor = self.get_data_transformer_object()
            
            
            preprocessor_obj= preprocessor.fit(input_feature_train_df)
            transformer_input_feature_train = preprocessor_obj.transform(input_feature_train_df)
            transformer_input_feature_test = preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[transformer_input_feature_train,np.array(target_feature_train_df)]
            test_arr=np.c_[transformer_input_feature_test,np.array(target_feature_test_df)]
            
            

        
            #saving numpy array
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_obj)
            
            
            logging.info("Saved transformed training and testing array")
            save_object("final_model/preprocessor.pkl",preprocessor_obj)
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_path=self.data_transformation_config.transformed_object_file_path
            )
            logging.info("Exited the initiate_data_transformation method of Data_Transformation class")
            return data_transformation_artifact
            
        except Exception as e:
            raise Bank_Exception(e, sys)    
        
        
        