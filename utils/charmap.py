import numpy as np

class CharMap:
    def __init__(self, chars):
        self.__onehot_mp, self.__order_mp = self.__create_maps(chars)

    def __create_maps(self, uchars):
        uchars = sorted(uchars)
        nchars = len(uchars)
        onehot_map = {}
        order_map = {}
        for i, c in enumerate(uchars):
            tmp = np.zeros(nchars)
            tmp[i] = 1
            onehot_map[c] = tmp
            order_map[c] = i
        return onehot_map, order_map

    def get_onehot(self, c):
        return self.__onehot_mp[c]

    def get_order(self, c):
        return self.__order_mp[c]

    def vocab_size(self):
        return len(self.__order_mp)
