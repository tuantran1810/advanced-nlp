import numpy as np
from text_generator import TextGeneratorModel
import torch
from torch import Tensor
from torch.nn import functional as F

if __name__ == "__main__":
    tg = TextGeneratorModel(100, 50, 256, 250)
    x = np.random.randint(0, 100, size = (32, 250))
    yhat = tg(Tensor(x).type(torch.long)).permute(0,2,1)

    print(yhat.shape)
    y = Tensor(np.random.randint(0, 100, size = (32, 250))).type(torch.long)

    loss = F.cross_entropy(yhat, y)
    print(loss)
    print(yhat.shape)
    yhat = F.log_softmax(yhat, dim = 1)
    acc = (torch.argmax(yhat, dim = 1) - y) == 0
    
    print(acc.sum())