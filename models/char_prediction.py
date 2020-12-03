from torch import nn

class CharPredictionModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_lstm):
        super(CharPredictionModel, self).__init__()
        self.__emb = nn.Embedding(vocab_size, emb_size)
        self.__lstm = nn.LSTM(emb_size, hidden_lstm)
        self.__drop = nn.Dropout(0.15)
        self.__fc = nn.Linear(hidden_lstm, vocab_size)

        self.__hidden_lstm = hidden_lstm

    def init_hidden(self, batch_size):
        return (
            torch.zeros(size = (1, batch_size, self.__hidden_lstm)),
            torch.zeros(size = (1, batch_size, self.__hidden_lstm))
        )

    def infer(self, inp, hidden):
        self.eval()
        return self.__forward(inp, hidden)

    def forward(self, x):
        x, _ = self.__forward(x, None)
        return x

    def __forward(self, x, hidden):
        batch_size = x.shape[0]
        x = self.__emb(x).transpose(0, 1)
        _, hidden = self.__lstm(x, hidden)
        x = hidden[0]
        x = x.view(batch_size, -1)
        x = self.__drop(x)
        x = self.__fc(x)
        return x, hidden
