
class Params:
    import os
    base_path = os.path.dirname(os.path.realpath(__file__))   
    # ------> Preprocessed Parameters
    data_files_format = "../DuReader/{mode}set/{type}.{mode}.json"
    train_files = ["../DuReader/trainset/{type}.train.json"]
    dev_files = ["../DuReader/devset/{type}.dev.json"]
    endline = None #100

    # ------> Vocabulary Parameters
    wordlst_path = "word_list.dict"   # save words' list
    vocab_path = base_path + "/vocab.dict"
    char_path = base_path + "/char.dict"
    count_threshold = 11     # We only save count

    # -----> Word Embedding Parameters
    epoch = 15
    vocabulary_size = 144836
    vocabulary_embedding = 386
    char_dict_size = 8136

    # -----> Generate Question-Answer Data Format Configuration


    struct_file = "./upload_res.json"

    
