## This file contain the raw code if i used the airflow  as the data ingestion part 


## @ A. Remove dotenv-based MySQL connection:
# from dotenv import load_dotenv
# load_dotenv()

# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_USER = os.getenv("MYSQL_USER")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
# MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# connection = mysql.connector.connect(
#     host=MYSQL_HOST,
#     user=MYSQL_USER,
#     password=MYSQL_PASSWORD,
#     database=MYSQL_DATABASE
# )
# cursor = connection.cursor()
# cursor.execute(f"SELECT * FROM {table_name}")
# rows = cursor.fetchall()
# df = pd.DataFrame(rows, columns=[col[0] for col in cursor.description])


# ## B B. Replace manual connection with Airflow’s MySqlHook
# from airflow.hooks.mysql_hook import MySqlHook

# def export_table_as_dataframe(self):
#     try:
#         hook = MySqlHook(mysql_conn_id="mysql_default")
#         df = hook.get_pandas_df(f"SELECT * FROM {self.data_ingestion_config.table_name}")
#         return df
#     except Exception as e:
#         raise Bank_Exception(e, sys)



## 2️⃣ Changes in config.py → Pipeline Configuration for Airflow

## Remove this 

# self.artifact_name = os.path.join(pipelineconstants.ARTIFACT_DIR, timestamp)
# self.artifact_dir = os.path.join(self.artifact_name, timestamp)

## change to 
# self.artifact_dir = os.path.join(pipelineconstants.ARTIFACT_DIR, timestamp)


##3️⃣ Changes in artifact.py → Minimal / No change
# @dataclass
# class DataIngestionArtifact:
#     train_file_path: str
#     test_file_path: str


# @dataclass
# class DataIngestionArtifact:
#     train_file_path: str
#     test_file_path: str
#     run_id: str = None
