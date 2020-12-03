from sklearn.model_selection import train_test_split
import os
import pickle
import torch
from loguru import logger as log
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from framework import Trainer
from utils import CharMap, unique_chars, split_xy_last_char
from models import CharPredictionModel

class CharInference:
    def __init__(self, datafile):
        (self.__x, self.__y), self.__char_map = self.__prepair_data(datafile)
        if not os.path.exists("./char-map"):
            os.mkdir("./char-map")
        with open("./char-map/char_inference.pkl", "wb") as fd:
            pickle.dump(self.__char_map, fd)

    def __prepair_data(self, files):
        def load_pkl_obj(file):
            with open(file, 'rb') as fd:
                return pickle.load(fd)
            return None

        all_chunks = list()
        all_chars = set()
        for f in files:
            obj = load_pkl_obj(f)
            all_chunks.extend(obj.allchunks)
            all_chars.update(obj.allchars)

        log.info(f"receive {len(all_chunks)} records, each has the length of {len(all_chunks[0])} chars, with {len(all_chars)} unique chars")
        char_map = CharMap(all_chars)
        return split_xy_last_char(all_chunks, char_map.get_order), char_map

    def train(self, epochs):
        def inject_data():
            x_train, x_test, y_train, y_test = train_test_split(self.__x, self.__y, train_size=0.9)
            x_train = Tensor(x_train).type(torch.long)
            x_test = Tensor(x_test).type(torch.long)
            y_train = Tensor(y_train).type(torch.long)
            y_test = Tensor(y_test).type(torch.long)

            traindataset = TensorDataset(x_train, y_train)
            testdataset = TensorDataset(x_test, y_test)
            traindataloader = DataLoader(traindataset, batch_size=256, shuffle=True)
            testdataloader = DataLoader(testdataset, batch_size=256, shuffle=False)

            return traindataloader, testdataloader

        def inject_model():
            return CharPredictionModel(self.__char_map.vocab_size(), 50, 512)

        def inject_optim(model):
            return optim.Adam(model.parameters(), lr=0.001)

        def inject_loss_fn():
            def loss(yhat, y):
                return F.cross_entropy(yhat, torch.flatten(y))
            return loss

        def inject_accuracy_calculator():
            def fn(yhat, y):
                yhat = F.log_softmax(yhat, dim = 1)
                tmp = (torch.argmax(yhat, dim = 1) - y) == 0
                return tmp.sum()/y.shape[0]
            return fn

        trainer = Trainer(
            "char-predictor",
            inject_data,
            inject_model,
            inject_optim,
            inject_loss_fn,
            inject_accuracy_calculator,
        )
        trainer.train(epochs)
        return trainer.get_model()

def main():
    char_inference = CharInference([
        "./data/pkl/books/dracula.txt",
        "./data/pkl/books/moby-dick.txt",
        "./data/pkl/books/pride-prejudice.txt",
        "./data/pkl/books/tale-of-two-cities.txt"
    ])
    char_inference.train(5)

if __name__ == "__main__":
    main()
