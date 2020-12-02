import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

class CharInference:
    def __init__(self, model, init_sentence, char_map):
        self.__model = model.cpu()
        self.__char_map = char_map
        nsentence = []
        for c in init_sentence:
            nsentence.append(char_map.get_order(c))
        init_sentence = Tensor(np.array(nsentence)).unsqueeze(0)
        self.__out, self.__hidden = model.infer(init_sentence)

    def __char_from_net_output(self, out):
        out = F.log_softmax(out, dim = 1)
        out = torch.argmax(out, dim = 1)
        return out, self.__char_map.char_from_order(out)
        
    def infer(self, nchars):
        n, c = self.__char_from_net_output(self.__out)
        lst = [c.numpy()[0]]
        for _ in range(nchars - 1):
            out, self.__hidden = self.__model.infer(n, self.__hidden)
            n, c = self.__char_from_net_output(out)
            lst.append(c)
