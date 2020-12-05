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
        self.__init_sentence = Tensor(np.array(nsentence)).unsqueeze(0).type(torch.long)

    def __char_from_net_output(self, out):
        out = F.softmax(out, dim = 1).numpy()
        out = out.flatten()
        nrange = len(out)
        out = np.random.choice(nrange, p = out)
        return out, self.__char_map.char_from_order(out)
        
    def infer(self, nchars):
        with torch.no_grad():
            out, hidden = self.__model.infer(self.__init_sentence, None)
            n, c = self.__char_from_net_output(out)
            lst = [c]
            for _ in range(nchars - 1):
                n = torch.Tensor([[n]]).type(torch.long)
                out, hidden = self.__model.infer(n, hidden)
                n, c = self.__char_from_net_output(out)
                lst.append(c)
        return lst
