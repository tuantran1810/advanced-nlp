from sklearn.model_selection import train_test_split
import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from framework import Trainer
from utils import CharMap, unique_chars, split_xy_last_char
from models import CharPredictionModel

def prepair_data(file):
    dataset = []
    with open(file, "r") as fd:
        for line in fd.readlines():
            line = line.strip('\n')
            dataset.append(line)

    uchars = unique_chars(dataset)
    char_map = CharMap(uchars)

    return split_xy_last_char(dataset, char_map.get_order), char_map

def main():
    (x, y), char_map = prepair_data("./name_data_char_sequences.txt")

    def inject_data():
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9)
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
        return CharPredictionModel(char_map.vocab_size(), 50, 512)

    def inject_optim(model):
        return optim.Adam(model.parameters(), lr=0.01)

    def inject_loss_fn():
        return F.cross_entropy

    def inject_accuracy_calculator():
        def fn(yhat, y):
            yhat = F.log_softmax(yhat, dim = 1)
            tmp = (torch.argmax(yhat, dim = 1) - y) == 0
            return tmp.sum()
        return fn

    trainer = Trainer(
        "char-predictor",
        inject_data,
        inject_model,
        inject_optim,
        inject_loss_fn,
        inject_accuracy_calculator,
    )
    trainer.train(5)

if __name__ == "__main__":
    main()
