"""
---Deprecated

Generate Vocabulary File,
And also generate a vocab list to train word-embedding.
"""
from __init__ import *
from params import Params


def _gen_vocab(datas):
    """
    It should be defined by diffent usage.
    :params
        datas is a json of data.
    :result
        put all paragraphs and questions' words into vocabulary dictionary.
    """
    for doc in datas["documents"]:
        [vocabulary.process_word_list(para) for para in doc["segmented_paragraphs"]]


def _gen_vocab_file(filepath, endline=None):
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:   logger.info("HAVE PROCESSED DONE %s LINES." % i)
            if i == endline: break
            _gen_vocab(json.loads(line))

def generate_vocabulary():
    for filepath in Params.train_files:
        _gen_vocab_file(filepath, Params.endline)


if __name__ == "__main__":
    generate_vocabulary() # First Generate Vocabulary and load character Dictionary
    vocabulary.save()
    
