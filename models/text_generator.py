import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_lstm, string_length = 250, hidden_fc = 200, lstm_layers = 2, batch_size = 32):
        super(TextGeneratorModel, self).__init__()
        self.__emb = nn.Embedding(vocab_size, emb_size)
        self.__lstm = nn.LSTM(emb_size, hidden_lstm, lstm_layers, bidirectional = True, batch_first = True, dropout = 0.1)
        self.__w = nn.Parameter(torch.randn(1, lstm_layers*hidden_lstm))
        self.__fc1 = nn.Linear(lstm_layers*hidden_lstm, hidden_fc)
        self.__fc2 = nn.Linear(hidden_fc, vocab_size)

    def infer(self, inp, hidden):
        self.eval()
        return self.__forward(inp, hidden)

    def forward(self, x):
        x, _ = self.__forward(x, None)
        return x

    def __forward(self, x, hidden):
        x = self.__emb(x)
        if hidden is not None:
            x, hidden = self.__lstm(x, hidden)
        else:
            x, hidden = self.__lstm(x)
        x = x.permute(0, 2, 1)
        m = torch.tanh(x)
        alpha = torch.matmul(self.__w, m).squeeze(1)
        alpha = F.softmax(alpha, dim = 1).unsqueeze(2)
        x = torch.matmul(x, alpha).squeeze(2)
        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        return x, hidden
