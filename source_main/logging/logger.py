import logging
import os
from datetime import datetime


LOGFILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGPATH = os.path.join(os.getcwd(),'log',LOGFILE)

os.makedirs(LOGPATH, exist_ok=True)

LOGFILEPATH=os.path.join(LOGPATH,LOGFILE)

logging.basicConfig(filename=LOGFILEPATH,
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

