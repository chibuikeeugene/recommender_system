import pandas as pd
from sklearn.preprocessing import LabelEncoder
from recommender_model_package.recommender_model_package.config.core import ENCODER_DIR, config
from recommender_model_package.recommender_model_package import __version__ as _version

import pickle


def serviceOptedFunction(*, dataset: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    # isolate the services columns
    target = dataset.iloc[:, 22:].idxmax(axis=1)

    sole = LabelEncoder()
    # call the label encoder fit_transform method on our raw_targets to obtain a numerical representation for each service name
    sole.fit(target)
    transformed_target = sole.transform(target)

    # create a new column known as service opted within the dataframe
    dataset['service_opted_for'] =  transformed_target

    return dataset, transformed_target


def userServiceEncoder(*, dataset: pd.DataFrame):
    """ encoder function for our user-service variables"""
    # Encode user_id and item_id
    u_encoder = LabelEncoder()
    dataset['ncodpers'] = u_encoder.fit_transform(dataset['ncodpers'])

    # save user encoder in a pkl file
    save_file_name = f"{config.apps_config.user_encoder}{_version}.pkl"
    save_path = ENCODER_DIR / save_file_name
    with open(save_path, "wb") as fout:
        pickle.dump(u_encoder, fout)

    s_encoder = LabelEncoder()
    dataset['service_opted'] = s_encoder.fit_transform(dataset['service_opted'])

    # save service encoder in a pkl file
    save_file_name = f"{config.apps_config.service_encoder}{_version}.pkl"
    save_path = ENCODER_DIR / save_file_name
    with open(save_path, "wb") as fout:
        pickle.dump(s_encoder, fout)

    return dataset
