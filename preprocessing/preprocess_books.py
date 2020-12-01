import pickle
from loguru import logger as log

class BookPreprocessor:
    def __init__(self, bookpath):
        self.__lines = list()
        self.__allchars = None
        self.__uniquechars = set()
        self.__chunks = None
        with open(bookpath, 'r') as fd:
            for line in fd.readlines():
                self.__lines.append(line)
        with open(bookpath, 'r') as fd:
            self.__allchars = fd.read()
            for c in self.__allchars:
                self.__uniquechars.add(c)
        log.info("Done loading the book from {}\n{} lines was read\n{} chars in file\n{} unique characters"\
            .format(bookpath, len(self.__lines), len(self.__allchars), len(self.__uniquechars)))

    @property
    def lines(self):
        return self.__lines

    @property
    def allchars(self):
        return self.__allchars

    @property
    def allchunks(self):
        return self.__chunks

    def to_chunk(self, chunklength, pkfile = ""):
        chunks = list()
        if chunklength > len(self.__allchars):
            log.error("chunk length is too big!")
            return chunks
        nchars = len(self.__allchars)
        for end in range(chunklength, nchars):
            start = end - chunklength
            chunks.append(self.__allchars[start:end])
        log.info("create {} chunks with the size of {} characters".format(len(chunks), chunklength))
        self.__chunks = chunks
        if len(pkfile) != 0:
            log.info("write {} chunks to pickle file {}".format(len(chunks), pkfile))
            with open(pkfile, 'wb') as fd:
                pickle.dump(self, fd)
        return chunks
