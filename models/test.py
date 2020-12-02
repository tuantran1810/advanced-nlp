import numpy as np
from text_generator import TextGeneratorModel
import torch
from torch import Tensor, optim
from torch.nn import functional as F

if __name__ == "__main__":
    cuda_tensor = lambda x: Tensor(x).type(torch.long).cuda()

    tg = TextGeneratorModel(100, 50, 256).cuda()
    for name, _ in tg.named_parameters():
        print(name)

    x = np.random.randint(0, 100, size = (32, 250))
    yhat = tg(cuda_tensor(x))

    print(yhat.shape)
    y = cuda_tensor(np.random.randint(0, 100, size = (32,)))

    loss = F.cross_entropy(yhat, y)
    print(loss)
    print(yhat.shape)
    yhat = F.log_softmax(yhat, dim = 1)
    acc = (torch.argmax(yhat, dim = 1) - y) == 0
    
    print(acc.sum())

    tg = TextGeneratorModel(100, 50, 256, batch_size = 1).cuda()
    x = cuda_tensor(np.random.randint(0, 100, size = (1, 250)))
    out, hidden = tg.infer(x, None)
    print(out)

    x = cuda_tensor(np.random.randint(0, 100, size = (1, 1)))
    out, _ = tg.infer(x, hidden)
    print(out)