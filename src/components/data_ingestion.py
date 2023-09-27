import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig():
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "data.csv")


class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 


    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or components")
        try:

            logging.info("Reading the data")
            df = pd.read_csv("rdata/stud.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)

            logging.info("Train test split")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("Inmgestion of the data is completed")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )



        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train, test = obj.initiate_data_ingestion()
    print(train, test)





        