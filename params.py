
class Params:
    
    # ------> Preprocessed Parameters
    data_files_format = "../DuReader/data/preprocessed/{mode}set/zhidao.{mode}.json"
    train_files = ["../DuReader/data/preprocessed/trainset/zhidao.train.json"]
    dev_files = ["../DuReader/data/preprocessed/devset/zhidao.dev.json"]
    endline = None #100

    # ------> Vocabulary Parameters
    wordlst_path = "word_list.dict"   # save words' list
    vocab_path = "./vocab.dict"
    char_path = "./char.dict"
    count_threshold = 11     # We only save count

    # -----> Word Embedding Parameters
    epoch = 15
    vocabulary_size = 144836
    vocabulary_embedding = 386
    char_dict_size = 8136

    # -----> Generate Question-Answer Data Format Configuration


    struct_file = "./upload_res.json"

    
