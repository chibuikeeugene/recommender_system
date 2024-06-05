import os,sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import pandas as pd
import typing as t
from loguru import logger
from config.core import ENCODER_DIR, config
from data_processing.data_manager import load_encoder, load_model
from recommender_model_package import __version__ as _version

# load the label encoder
service_encoder_file_name = f"{config.apps_config.service_encoder}{_version}.pkl"
se = load_encoder(file_name=service_encoder_file_name)

# load the model
model_file_name = f"{config.apps_config.model_file_name}{_version}.pkl"
model = load_model(file_name=model_file_name)


def get_recommendation(*, uid: float,  
                       service_range: int):
    """ call this method to retrieve recommendations"""    

    # Get service names
    recommendations = [(uid, 
                        sid, 
                        se.inverse_transform([sid])[0], 
                        model.predict(uid, sid).est) for sid in range(service_range)]
    # Convert to pandas dataframe
    recommendations = pd.DataFrame(recommendations, columns=['uid', 'sid', 'service_name', 'pred'])
    # Sort by pred
    recommendations.sort_values("pred", ascending=False, inplace=True)
    # Reset index
    recommendations.reset_index(drop=True, inplace=True)

    logger.info("Recommendations for user: {} is ready!".format(uid))

    result = {value:key for key, value in config.models_config.feature_dict.items()}
    result_class =  [result[i] for i in recommendations.service_name]

    # Return recommendations
    logger.info(f"{service_range} recommended services for user {uid} are {result_class}")
  
    return result_class

    

