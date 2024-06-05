import os,sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from surprise import accuracy, Reader, Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from recommender_model_package.data_processing import data_manager as dm
from config.core import config




def run_training() -> None:
    """ key function to train the model"""

    # load the data
    loaded_data = dm.load_dataset(filename=config.apps_config.training_data)

    # transform and prepare the dataset
    user_item_ratio_stacked = dm.transform_dataset(dataset=loaded_data)


    # Creating surprise processable dataset
    # Initialize a surprise reader object
    reader = Reader(line_format='user item rating', sep=', ', rating_scale=(0, 1), skip_lines=1)

    # Load the data
    data = Dataset.load_from_df(user_item_ratio_stacked, reader=reader)

    trainset =  data.build_full_trainset()

    svd = SVD(n_epochs = config.models_config.n_epochs,
              lr_all = config.models_config.lr_all,
              reg_all = config.models_config.reg_all)

    svd.fit(trainset)
    dm.save_model(model=svd)

if __name__ == "__main__":
    run_training()