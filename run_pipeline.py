import os
import src.config as Config
from src.data_loader import DataLoader
from src.model import Classifier
from src.optimizer import Optimizer

def main():
    # create logging directory if it doesn't exist
    if not os.path.exists(Config.LOGGING_DIR):
        os.makedirs(Config.LOGGING_DIR)

    # load data
    data_loader = DataLoader(**Config.DATA_LOADER_CONFIG)
    print(f"Number of classes: {data_loader.get_num_classes()}")
    train_data, test_data = data_loader.get_train_data(), data_loader.get_test_data()

    # optimize hyperparameters
    optimizer = Optimizer(data_loader)
    best_params = optimizer.tune_hyperparams(n_trials=1)
    
    # train and evaluate the model
    clf = Classifier(**best_params, objective='multi:softmax', eval_metric='mlogloss')
    clf.train(train_data, data_loader.sample_weights)
    clf.evaluate(test_data)


if __name__ == "__main__":
    main()
