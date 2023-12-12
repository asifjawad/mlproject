import sys
import os 
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """
        This function is responsible for data transformation

        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Numerical Information {numerical_columns}")
            logging.info(f"Categorical Information {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("cat_pipelines",cat_pipeline,categorical_columns),
                ("num_pipeline",num_pipeline,numerical_columns)


                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_transformation(self, train_data, test_data):
        try:

            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            logging.info("Reading Train and Test data completed")

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = train_df[target_column_name]

            logging.info("Obtaining Preprocessing obj")
            preprocessing_obj = self.get_data_transformer_object()
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            print("shape of train",input_feature_test_arr.shape[1])


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            
            # save_object(

            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj

            # )


            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        




if __name__ == "__main__":
    transform = DataTransformation()
    transform.initiate_data_transformation("artifacts/train.csv","artifacts/test.csv")
    print("Your Pickle file is Ready")