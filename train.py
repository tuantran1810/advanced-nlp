import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor, cuda, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from utils import CharMap, unique_chars, split_xy_last_char
from models import CharPredictionModel

class CharPredictionTrainer:
    def __init__(self, x, y, vocal_size, emb_size, hidden_lstm, model_name = "char_prediction", learning_rate = 0.01, batch_size = 256):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model = CharPredictionModel(vocal_size, emb_size, hidden_lstm).to(self.__device)

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9)
        x_train = Tensor(x_train).type(torch.long)
        x_test = Tensor(x_test).type(torch.long)
        y_train = Tensor(y_train).type(torch.long)
        y_test = Tensor(y_test).type(torch.long)

        traindataset = TensorDataset(x_train, y_train)
        testdataset = TensorDataset(x_test, y_test)
        self.__traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
        self.__testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

        self.__optim = optim.Adam(self.__model.parameters(), lr=learning_rate)
        self.__batch_size = batch_size
        self.__model_name = model_name

    def train(self, epochs):
        for i in range(epochs):
            print(f"---------------------------epoch {i}-------------------------")
            self.__model.train()
            totalloss = 0
            cnt = 0
            for x, y in self.__traindataloader:
                cnt += 1
                x = x.to(self.__device)
                y = y.to(self.__device)

                self.__optim.zero_grad()
                yhat = self.__model(x)
                loss = F.cross_entropy(yhat, torch.flatten(y))
                loss.backward()
                self.__optim.step()
                with torch.no_grad():
                    totalloss += loss
            print(f"training loss: {totalloss/cnt}")
            torch.save(self.__model.state_dict, self.__model_name + ".pth")

            self.__model.eval()
            with torch.no_grad():
                totalloss = 0
                totalaccuracy = 0
                cnt = 0

                for x, y in self.__testdataloader:
                    cnt += 1
                    x = x.to(self.__device)
                    y = y.to(self.__device)
                    yhat = self.__model(x)
                    loss = F.cross_entropy(yhat, torch.flatten(y))
                    totalloss += loss
                    yhat = F.log_softmax(yhat, dim = 1)
                    tmp = (torch.argmax(yhat, dim = 1) - y) == 0
                    totalaccuracy += tmp.sum()

                print(f"loss = {totalloss/cnt}, accuracy = {(totalaccuracy/(cnt * self.__batch_size)) * 100}%")

def main():
    dataset = []
    with open("./name_data_char_sequences.txt", "r") as fd:
        for line in fd.readlines():
            line = line.strip('\n')
            dataset.append(line)

    uchars = unique_chars(dataset)
    char_map = CharMap(uchars)

    x, y = split_xy_last_char(dataset, char_map.get_order)
    trainer = CharPredictionTrainer(x, y, char_map.vocab_size(), 50, 512)
    trainer.train(5)

if __name__ == "__main__":
    main()
