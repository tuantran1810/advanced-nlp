import torch
from torch import nn
import torch.nn.functional as F

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_lstm, string_length = 250, hidden_fc = 4, lstm_layers = 2, batch_size = 32):
        super(TextGeneratorModel, self).__init__()
        self.__emb = nn.Embedding(vocab_size, emb_size)
        self.__lstm = nn.LSTM(emb_size, hidden_lstm, lstm_layers, bidirectional = True, batch_first = True, dropout = 0.1)
        self.__drop1 = nn.Dropout(0.1)
        self.__fc1 = nn.Linear(lstm_layers*hidden_lstm, hidden_fc)
        self.__drop2 = nn.Dropout(0.1)
        self.__fc2 = nn.Linear(string_length*hidden_fc, vocab_size)
        self.__drop3 = nn.Dropout(0.1)

        self.__batch_size = batch_size
        self.__lstm_layers = lstm_layers
        self.__hidden_lstm = hidden_lstm

    def init_hidden(self):
        return (
            torch.zeros(self.__lstm_layers*2, self.__batch_size, self.__hidden_lstm),
            torch.zeros(self.__lstm_layers*2, self.__batch_size, self.__hidden_lstm),
        )

    def infer(self, inp, hidden):
        self.eval()
        batch_size = inp.shape[0]
        return self.__forward(inp, hidden, batch_size)

    def forward(self, x):
        x, _ = self.__forward(x, None, self.__batch_size)
        return x

    def __forward(self, x, hidden, batch_size):
        x = self.__emb(x)
        if hidden is not None:
            x, hidden = self.__lstm(x, hidden)
        else:
            x, hidden = self.__lstm(x)
        x = self.__drop1(x)
        x = F.relu(self.__fc1(x))
        x = x.view(batch_size, -1)
        x = self.__drop2(x)
        x = F.relu(self.__fc2(x))
        x = self.__drop3(x)
        return x, hidden
