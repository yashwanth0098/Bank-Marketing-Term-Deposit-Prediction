import logging
import os
from datetime import datetime


LOGFILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGDIR  = os.path.join(os.getcwd(),'log',LOGFILE)

os.makedirs(LOGDIR, exist_ok=True)

LOGFILEPATH=os.path.join(LOGDIR,LOGFILE)

logging.basicConfig(filename=LOGFILEPATH,
                    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    )

