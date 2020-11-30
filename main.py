from execute import CharInference

def main():
    char_inference = CharInference("./data/name_data_char_sequences.txt")
    char_inference.train(5)

if __name__ == "__main__":
    main()
