## .env
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net
MONGO_DATABASE=bankdb
MONGO_COLLECTION=transactions



# data_ingestion_config.py
 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    mongo_uri: str
    database_name: str
    collection_name: str
    
    
#data_ingestion_artifact.py

from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionArtifact:
    data_frame: pd.DataFrame
    collection_name: str
    database_name: str
    status: str

###Data Ingestion 

import sys
import pandas as pd
import pymongo
from dotenv import load_dotenv
import os
from source_main.entity.config import DataIngestionConfig
from source_main.entity.artifact import DataIngestionArtifact
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging

load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Bank_Exception(e, sys)

    def export_collection_as_dataframe(self) -> DataIngestionArtifact:
        try:
            logging.info("Connecting to MongoDB...")
            client = pymongo.MongoClient(self.data_ingestion_config.mongo_uri)
            db = client[self.data_ingestion_config.database_name]
            collection = db[self.data_ingestion_config.collection_name]

            logging.info(f"Fetching data from collection: {self.data_ingestion_config.collection_name}")
            data = list(collection.find())
            df = pd.DataFrame(data)

            artifact = DataIngestionArtifact(
                data_frame=df,
                collection_name=self.data_ingestion_config.collection_name,
                database_name=self.data_ingestion_config.database_name,
                status="SUCCESS"
            )

            logging.info(f"Data fetched successfully: {df.shape}")
            return artifact

        except Exception as e:
            raise Bank_Exception(e, sys)
        
        
        
        
**2 .PostgreSQL Version**     
###data_ingestion_config.py

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    host: str
    user: str
    password: str
    database_name: str
    table_name: str

#data_ingestion_artifact.py

from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionArtifact:
    data_frame: pd.DataFrame
    table_name: str
    database_name: str
    status: str


## 🔹 data_ingestion.py
import sys
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from source_main.entity.config import DataIngestionConfig
from source_main.entity.artifact import DataIngestionArtifact
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging

load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Bank_Exception(e, sys)

    def export_table_as_dataframe(self) -> DataIngestionArtifact:
        try:
            logging.info("Connecting to PostgreSQL...")
            conn = psycopg2.connect(
                host=self.data_ingestion_config.host,
                user=self.data_ingestion_config.user,
                password=self.data_ingestion_config.password,
                dbname=self.data_ingestion_config.database_name
            )

            query = f"SELECT * FROM {self.data_ingestion_config.table_name}"
            df = pd.read_sql(query, conn)

            artifact = DataIngestionArtifact(
                data_frame=df,
                table_name=self.data_ingestion_config.table_name,
                database_name=self.data_ingestion_config.database_name,
                status="SUCCESS"
            )

            logging.info(f"Data fetched successfully from {self.data_ingestion_config.table_name}")
            return artifact

        except Exception as e:
            raise Bank_Exception(e, sys)
        
        
##Oracle DB – Enterprise Database

# data_ingestion_config.py

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    host: str
    port: int
    user: str
    password: str
    service_name: str
    table_name: str


# data_ingestion_artifact.py

from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionArtifact:
    data_frame: pd.DataFrame
    table_name: str
    database_service: str
    status: str
    
    
# data_ingestion.py

import sys
import pandas as pd
import oracledb
from dotenv import load_dotenv
import os
from source_main.entity.config import DataIngestionConfig
from source_main.entity.artifact import DataIngestionArtifact
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging

load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise Bank_Exception(e, sys)

    def export_table_as_dataframe(self) -> DataIngestionArtifact:
        try:
            dsn = oracledb.makedsn(
                self.config.host,
                self.config.port,
                service_name=self.config.service_name
            )

            logging.info(f"Connecting to Oracle DB: {self.config.service_name}")
            conn = oracledb.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=dsn
            )

            query = f"SELECT * FROM {self.config.table_name}"
            df = pd.read_sql(query, conn)

            artifact = DataIngestionArtifact(
                data_frame=df,
                table_name=self.config.table_name,
                database_service=self.config.service_name,
                status="SUCCESS"
            )
            logging.info(f"Data fetched successfully from {self.config.table_name}")
            return artifact

        except Exception as e:
            raise Bank_Exception(e, sys)

# #4 SQL Server (T-SQL / Microsoft SQL Server)
# data_ingestion_config.py


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    host: str
    user: str
    password: str
    database_name: str
    table_name: str

# data_ingestion_artifact.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionArtifact:
    data_frame: pd.DataFrame
    table_name: str
    database_name: str
    status: str

# data_ingestion.py
import sys
import pandas as pd
import pyodbc
from dotenv import load_dotenv
import os
from source_main.entity.config import DataIngestionConfig
from source_main.entity.artifact import DataIngestionArtifact
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging

load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise Bank_Exception(e, sys)

    def export_table_as_dataframe(self) -> DataIngestionArtifact:
        try:
            logging.info("Connecting to SQL Server...")
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config.host};"
                f"DATABASE={self.config.database_name};"
                f"UID={self.config.user};PWD={self.config.password}"
            )
            conn = pyodbc.connect(conn_str)

            query = f"SELECT * FROM {self.config.table_name}"
            df = pd.read_sql(query, conn)

            artifact = DataIngestionArtifact(
                data_frame=df,
                table_name=self.config.table_name,
                database_name=self.config.database_name,
                status="SUCCESS"
            )

            logging.info(f"Data fetched successfully from {self.config.table_name}")
            return artifact

        except Exception as e:
            raise Bank_Exception(e, sys)

# 5️⃣ AWS S3 (boto3)
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    file_key: str
    file_type: str = "csv"  # or parquet/json


# data_ingestion_artifact.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionArtifact:
    data_frame: pd.DataFrame
    bucket_name: str
    file_key: str
    status: str


# data_ingestion.py
import sys
import pandas as pd
import boto3
import io
from dotenv import load_dotenv
import os
from source_main.entity.config import DataIngestionConfig
from source_main.entity.artifact import DataIngestionArtifact
from source_main.exception.exception import Bank_Exception
from source_main.logging.logger import logging

load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise Bank_Exception(e, sys)

    def export_file_as_dataframe(self) -> DataIngestionArtifact:
        try:
            logging.info("Connecting to AWS S3...")
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key
            )

            obj = s3.get_object(Bucket=self.config.bucket_name, Key=self.config.file_key)
            if self.config.file_type.lower() == "csv":
                df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            elif self.config.file_type.lower() == "parquet":
                df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            elif self.config.file_type.lower() == "json":
                df = pd.read_json(io.BytesIO(obj['Body'].read()))
            else:
                raise ValueError("Unsupported file type")

            artifact = DataIngestionArtifact(
                data_frame=df,
                bucket_name=self.config.bucket_name,
                file_key=self.config.file_key,
                status="SUCCESS"
            )

            logging.info(f"File {self.config.file_key} fetched successfully from S3")
            return artifact

        except Exception as e:
            raise Bank_Exception(e, sys)

# 6️⃣ Airflow DAG Integration
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from source_main.components.data_ingestion import DataIngestion
from source_main.entity.config import DataIngestionConfig
import os

def run_postgres_ingestion():
    config = DataIngestionConfig(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database_name=os.getenv("DB_NAME"),
        table_name="transactions"
    )
    DataIngestion(config).export_table_as_dataframe()

with DAG(
    dag_id="data_ingestion_pipeline",
    description="Daily ingestion pipeline for data warehouse",
    start_date=datetime(2025, 10, 20),
    schedule_interval="@daily",
    catchup=False,
    tags=["ingestion", "etl"]
) as dag:
    ingestion_task = PythonOperator(
        task_id="postgres_ingestion_task",
        python_callable=run_postgres_ingestion
    )

