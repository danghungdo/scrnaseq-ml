import os
import src.config as Config
from src.data_loader import DataLoader
from src.model import Classifier


def main():
    # create logging directory if it doesn't exist
    if not os.path.exists(Config.LOGGING_DIR):
        os.makedirs(Config.LOGGING_DIR)

    # load data
    data_loader = DataLoader(**Config.DATA_LOADER_CONFIG)
    train_data, test_data = data_loader.get_train_data(), data_loader.get_test_data()

    # train model and evaluate
    clf = Classifier(**Config.MODEL_CONFIG)
    clf.train(train_data)
    clf.evaluate(test_data)


if __name__ == "__main__":
    main()
