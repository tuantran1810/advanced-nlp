from torch import nn

class CharPredictionModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_lstm):
        super(CharPredictionModel, self).__init__()
        self.__emb = nn.Embedding(vocab_size, emb_size)
        self.__lstm = nn.LSTM(emb_size, hidden_lstm)
        self.__drop = nn.Dropout(0.15)
        self.__fc = nn.Linear(hidden_lstm, vocab_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.__emb(x).transpose(0, 1)
        _, (x, _) = self.__lstm(x)
        x = x.view(batch_size, -1)
        x = self.__drop(x)
        x = self.__fc(x)
        return x
