
from copy import deepcopy as copy
def padding_data(data, max_len):
    t = copy(data)
    t.extend([0] * (max_len - len(data)))
    return t


def padding_char_data(data, max_len, max_clen):
    t = copy(data)
    for p in t:
        p.extend([0] * (max_clen - len(p)))
    t.extend( [[0] * max_clen] * (max_len - len(t)))
    return t


def padding_char_len(data, max_len):
    t = copy(data)
    for p in t:
        p.extend([0] * (max_len - len(p)))
    return t

def padding_data(data, max_len):
    data.extend([0] * (max_len - len(data)))
    return data

def padding_char_data(data, max_len, max_clen):
    for p in data:
        p.extend([0] * (max_clen - len(p)))
    data.extend( [[0] * max_clen] * (max_len - len(data)))
    return data


def padding_char_len(data, max_len):
    for p in data:
        p.extend([0] * (max_len - len(p)))
    return data

import json, numpy as np
def ljz_load_data(_dir=None):
    _file = "../../Zeus/description_id.stat"  # description
    passage_word_ids, question_word_ids = [], []
    passage_char_ids, question_char_ids = [], []
    passage_word_len, question_word_len = [], []
    passage_char_len, question_char_len = [], []
    indices = []

    max_plen, max_qlen, max_clen = Params.max_p_len, Params.max_q_len, Params.max_char_len

    with open(_file) as fp:
        for i, line in enumerate(fp):

            if i == 100: break
            d = json.loads(line)
            if len(d["segmented_paragraph"]) > max_plen or len(d["segmented_question"]) > max_qlen:
                print(len(d["segmented_paragraph"]), len(d["segmented_question"]))
                continue

            passage_word_ids.append( padding_data(d["segmented_paragraph"], max_plen))
            question_word_ids.append( padding_data(d["segmented_question"], max_qlen))
            passage_char_ids.append( padding_char_data(d["char_paragraph"], max_plen, max_clen))
            question_char_ids.append( padding_char_data(d["char_question"], max_qlen, max_clen))

            passage_word_len.append([len(d["segmented_paragraph"])])
            question_word_len.append([len(d["segmented_question"])])
            passage_char_len.append([len(word) for word in d["char_paragraph"]])
            question_char_len.append([len(word) for word in d["char_question"]])
            indices.append(d["answer_spans"])

        # to numpy
        indices = np.reshape(np.asarray(indices,np.int32),(-1,2))
        passage_word_len = np.reshape(np.asarray(max_plen, np.int32),(-1,1))
        question_word_len = np.reshape(np.asarray(max_qlen, np.int32),(-1,1))
        # p_char_len = pad_data(p_char_len,p_max_word)
        # q_char_len = pad_data(q_char_len,q_max_word)
        p_char_len = padding_char_len(passage_char_len, max_plen)
        q_char_len = padding_char_len(question_char_len, max_qlen)
        # shapes of each data
        shapes=[(max_plen,),(max_qlen,),
                (max_plen, max_clen,),(max_qlen,max_clen,),
                (1,),(1,),
                (max_plen,),(max_qlen,),
                (2,)]

        return ([np.array(passage_word_ids), np.array(question_word_ids),
                np.array(passage_char_ids), np.array(question_char_ids),
                passage_word_len, question_word_len,
                p_char_len, q_char_len,
                indices], shapes)

if __name__ == "__main__":
    a, b = get_batch()

    max_p_len = 500
    max_q_len = 20
    max_char_len = 8
    vocab_size = 469228
    char_vocab_size = 5160
    emb_size = 300
    char_emb_size = 100
