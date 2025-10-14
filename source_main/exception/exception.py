from source_main.logging import logger
import sys

class Bank_Exception(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message=error_message
        _,_,exc_tb=error_details.exc_info()
        self.file_name=exc_tb.tb_frame.f_code.co_filename
        self.line_number=exc_tb.tb_lineno
        
        def __str__(self):
            return "Error occured in script:[{0}] at line number:[{1}] error message:[{2}]".format(
                self.file_name,self.line_number,str(self.error_message))
        
if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        bank_exception=Bank_Exception(e,sys)
        logger.logging.info(bank_exception.error_message)
        raise bank_exception