# import os,sys

# sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))

import typing as t
from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import numpy as np
import recommender_model_package.recommender_model_package.data_processing.preprocessor as pp
from recommender_model_package.recommender_model_package.config.core import config, SANTANDER_DATA_DIR, TRAINED_MODEL_DIR, ENCODER_DIR
from recommender_model_package.recommender_model_package import __version__ as _version

import pickle
import joblib


def load_dataset(*, filename: str) -> pd.DataFrame:
    """ function to read dataset"""
    logger.info(f"Loading file {filename} ...")
    train_data = pd.read_csv(Path(f"{SANTANDER_DATA_DIR}/{filename}"), engine='c')
    logger.info(f"shape of the loaded dataset is: {train_data.shape}")

    logger.info("Removing unnecessary columns ('ult_fec_cli_1t', 'conyuemp') ...")
    train_data = train_data.drop(labels=['ult_fec_cli_1t', 'conyuemp'], axis=1)

    logger.info(f"New shape of the dataset: {train_data.shape}")

    return train_data


def transform_dataset(*, dataset: pd.DataFrame) -> pd.DataFrame:
    """ function to transform our dataset for training"""
    # call the preprocessor function
    logger.info("Collating all service columns and creating a new feature that represents the top rated services for each user ...")
    train_data,transformed_target = pp.serviceOptedFunction(dataset=dataset)

    
    # Creating a user-item matrix, each entry indicates the number of times service opted by that user
    logger.info("Generating a user-service matrix ...")
    user_item_matrix = pd.crosstab(index=train_data.ncodpers, columns=transformed_target, values=1, aggfunc='sum')

    # Filling nan values as 0 as service is not opted
    logger.info("addressing any missing values in our user-item matrix ...")
    user_item_matrix.fillna(0, inplace=True)


    # Having calculated the number of times a user has opted for a service. Then for each user we will divide the count of 
    # each service with the total number of services the user has opted throughout his/her banking journey.
    # Convert the user_item_matrix to array datatype
    uim_arr = np.array(user_item_matrix)

    # Iterate through each row(user)
    for row,item in tqdm(enumerate(uim_arr)):
        # Iterate through each column(item)
        for column,item_value in enumerate(item):
            # Change the count of service opted to ratio
            uim_arr[row, column] = uim_arr[row, column] / sum(item)
            
    # Convert the array to dataframe for better view
    user_item_ratio_matrix = pd.DataFrame(uim_arr, columns=user_item_matrix.columns, index=user_item_matrix.index)

    # Stack the user_item_ratio_matrix to get all values in single column
    user_item_ratio_stacked = user_item_ratio_matrix.stack().to_frame()

    # Create column for user id
    user_item_ratio_stacked['ncodpers'] = [index[0] for index in user_item_ratio_stacked.index]

    # Create column for service_opted
    user_item_ratio_stacked['service_opted'] = [index[1] for index in user_item_ratio_stacked.index]

    # Reset and drop the index
    user_item_ratio_stacked.reset_index(drop=True, inplace=True)



    # Formating our final dataset

    # Rename the column 0 to service_selection_ratio
    user_item_ratio_stacked.rename(columns={0:"service_selection_ratio"}, inplace=True)

    # Arange the column systematicaly for better view
    user_item_ratio_stacked = user_item_ratio_stacked[['ncodpers','service_opted', 'service_selection_ratio']]

    # Drop all the rows with 0 entries as it means the user has never opted for the service
    user_item_ratio_stacked.drop(user_item_ratio_stacked[user_item_ratio_stacked['service_selection_ratio']==0].index, inplace=True)

    # Reset the index
    user_item_ratio_stacked.reset_index(drop=True, inplace=True)

    logger.info("Encoding the user and service columns...")
    final_transformed_dataset = pp.userServiceEncoder(dataset=user_item_ratio_stacked)

    # view the first few rows of the dataset
    logger.log(_Logger__message = final_transformed_dataset.head(), _Logger__level = 2)
    return final_transformed_dataset

def save_model(*, model: t.Any) -> None:
    """ function to save the model"""
    logger.info("Saving the model ...")
    save_file_name = f"{config.apps_config.model_file_name}{_version}.bin"
    save_path = TRAINED_MODEL_DIR / save_file_name
    with open(save_path, "wb") as fout:
        joblib.dump(model, fout, compress=3)
    logger.log(_Logger__message = model, _Logger__level = 2)


def load_model(*, file_name: str) -> t.Any:
    """ function to load the model"""
    logger.info("Loading the model ...")
    load_path = TRAINED_MODEL_DIR / file_name
    with open(load_path, "rb") as fin:
        model = joblib.load(fin)
    logger.log(_Logger__message = model, _Logger__level = 2)
    return model


def load_encoder(*, file_name: str) -> LabelEncoder:
    """ function to load the encoder"""
    logger.info("Loading the encoder ...")
    load_path = ENCODER_DIR / file_name
    with open(load_path, "rb") as fin:
        encoder = pickle.load(fin)
    logger.log(_Logger__message = encoder, _Logger__level = 2)
    return encoder