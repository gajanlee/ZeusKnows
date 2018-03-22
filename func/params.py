class Params:
    
    # ------> Preprocessed Parameters
    train_files = ["/home/lijz/Dureader/data/preprocessed/trainset/zhidao.train.json"]
    endline = None

    # ------> Vocabulary Parameters
    wordlst_path = "word_list.dict"   # save words' list
    vocab_path = "./vocab.dict"
    char_path = "./char.dict"
    count_threshold = 11     # We only save count

    # -----> Word Embedding Parameters
    epoch = 15

    
