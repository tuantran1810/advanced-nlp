from preprocess_books import BookPreprocessor

def test_book_processor():
    p = BookPreprocessor("../data/books/dracula.txt")
    p.to_chunk(250, "test.pkl")

if __name__ == "__main__":
    test_book_processor()
