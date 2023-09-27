import sys
from src.logger import logging



def error_message_detail(error,error_detail:sys):
    _,_,tb = error_detail.exc_info()
    filename = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    error_message = "Error occured in [{0}], at line number [{1}], with error message [{2}]".format(
        filename,line_number,str(error)
        )

    return error_message




class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message





if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Logging has started")
        raise CustomException(e,sys)