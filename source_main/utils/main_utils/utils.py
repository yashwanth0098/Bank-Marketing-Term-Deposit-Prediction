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





def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'rb') as yaml_file:
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