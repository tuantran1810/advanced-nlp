from loguru import logger as log
from preprocessing import BookPreprocessor

def main():
    books = [
        "dracula.txt",
        "moby-dick.txt",
        "pride-prejudice.txt",
        "tale-of-two-cities.txt"
    ]
    bookspath = "./data/books/"
    pklpath = "./data/pkl/books/"

    log.info("start processing for {} books".format(len(books)))
    for book in books:
        log.info("processing for the book: {}".format(book))
        BookPreprocessor(bookspath + book).to_chunk(250, pklpath + book)

if __name__ == "__main__":
    main()
