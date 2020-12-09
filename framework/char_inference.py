import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

class CharInference:
    def __init__(self, model, init_sentence, char_map):
        self.__model = model.cpu()
        self.__char_map = char_map
        self.__prob_threshold = 1.0/char_map.vocab_size()
        self.__init_sentence = self.__build_sentence_tensor(init_sentence)

    def __build_sentence_tensor(self, sentence):
        nsentence = list()
        for c in sentence:
            nsentence.append(self.__char_map.get_order(c))
        return Tensor(np.array(nsentence)).unsqueeze(0).type(torch.long)

    def __char_from_net_output(self, out):
        out = F.softmax(out, dim = 1).numpy()
        out = out.flatten()
        nrange = len(out)
        out = np.random.choice(nrange, p = out)
        return out, self.__char_map.char_from_order(out)

    def __char_from_num_list(self, lst):
        ret = list()
        for n in lst:
            c = self.__char_map.char_from_order(n)
            ret.append(c)
        return ret

    def __mostlikely_nchar(self, prob, norigchar):
        origchar = self.__char_map.char_from_order(norigchar)
        isalnum = origchar.isalnum()
        nchar_lst = np.argsort(prob)[::-1]
        for n in nchar_lst:
            c = self.__char_map.char_from_order(n)
            if c.isalnum() == isalnum:
                return n, prob[n]
        return None, None

    def __possible_chars(self, arr, norigchar):
        npout = F.softmax(arr).numpy().flatten()
        mask = npout > self.__prob_threshold
        nchars_possible = npout * mask
        nchars_possible = np.argwhere(nchars_possible > 0).flatten()
        nchar, confidence = self.__mostlikely_nchar(npout, norigchar)
        return self.__char_from_num_list(nchars_possible), self.__char_map.char_from_order(nchar), confidence

    def spellcheck(self, sentence):
        nsentence = self.__build_sentence_tensor(' ' + sentence).numpy().flatten()
        mistake = {}
        with torch.no_grad():
            out, hidden = self.__model.infer(self.__init_sentence, None)
            for i in range(len(sentence)):
                n = nsentence[i]
                n = torch.Tensor(np.array([n])).unsqueeze(0).type(torch.long)
                out, hidden = self.__model.infer(n, hidden)
                pchars, recommend, confident = self.__possible_chars(out, nsentence[i + 1])
                if sentence[i] not in pchars:
                    mistake[i] = (recommend, confident)
                    nsentence[i + 1] = self.__char_map.get_order(recommend)
        return mistake

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
