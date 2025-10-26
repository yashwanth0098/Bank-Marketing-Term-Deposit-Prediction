from source_main.constants.pipelineconstants import MODEL_FILE_NAME,SAVE_MODEL_DIR
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging
import os
import sys

class BankModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise Bank_Exception(e,sys)
        
    def predict(self,x):
        try:
            x_transform=self.preprocessor.transform(x)
            y_hat=self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise Bank_Exception(e,sys)