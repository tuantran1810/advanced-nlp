import torch
from torch import nn

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_lstm, sequence_length, lstm_layers = 2, batch_size = 32):
        super(TextGeneratorModel, self).__init__()
        self.__emb = nn.Embedding(vocab_size, emb_size)
        self.__lstm = nn.LSTM(emb_size, hidden_lstm, lstm_layers, bidirectional = True, batch_first = True)
        self.__drop = nn.Dropout(0.1)
        self.__fc1 = nn.Linear(lstm_layers*hidden_lstm, vocab_size)
        self.__fc2 = nn.Linear(vocab_size, vocab_size)
        self.__hidden = self.__init_hidden(lstm_layers, batch_size, hidden_lstm)

    def __init_hidden(self, lstm_layers, batch_size, hidden_lstm):
        return (
            torch.zeros(lstm_layers*2, batch_size, hidden_lstm),
            torch.zeros(lstm_layers*2, batch_size, hidden_lstm),
        )

    def infer(self, inp, hidden = None):
        self.eval()
        if hidden is None:
            hidden = self.__hidden
        return self(inp, hidden)

    def forward(self, x):
        x = self.__emb(x)
        x, self.__hidden = self.__lstm(x)
        x = self.__drop(x)
        x = self.__fc1(x)
        x = self.__fc2(x)
        return x
