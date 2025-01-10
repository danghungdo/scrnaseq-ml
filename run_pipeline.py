import os
import pickle
import numpy as np
import src.config as Config

from src.data_loader import DataLoader
from src.model import Classifier
from src.optimizer import Optimizer


from src.evaluation_metrics import EvaluationMetrics



def main():
    # set the seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    # create logging directory if it doesn't exist
    if not os.path.exists(Config.LOGGING_DIR):
        os.makedirs(Config.LOGGING_DIR)

    # load dataloader
    path_dataloader = "data/dataloader.pkl"
    if not os.path.exists(path_dataloader):
        data_loader = DataLoader(**Config.DATA_LOADER_CONFIG, random_seed=Config.RANDOM_SEED)
        print(f"Number of classes: {data_loader.get_num_classes()}")
        # save dataloader
        with open(path_dataloader, "wb") as output:
            pickle.dump(data_loader, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(path_dataloader, "rb") as input:  # 'rb' for reading binary
            data_loader = pickle.load(input)

    # load train, test data
    train_data, test_data = (
        data_loader.get_train_data(),
        data_loader.get_test_data(),
    )
    # optimize hyperparameters
    optimizer = Optimizer(data_loader)
    best_params = optimizer.tune_hyperparams(n_trials=100)

    # train the model
    clf = Classifier(**best_params, objective="multi:softmax", eval_metric="mlogloss")
    clf.train(train_data, data_loader.sample_weights)

    # evaluate the model
    eval_metrics = EvaluationMetrics(model=clf, data_loader=data_loader)
    eval_metrics.evaluate()

if __name__ == "__main__":
    main()
