"""
---Deprecated

Convert Word2vec Output File to a json file (vocab.dict/ char.dict). 
"""

import word2vec
import json
import numpy as np
model = word2vec.load("wordsVec2.bin")
print("load Model Over")
vocab_dict, char_dict = {"<vocab-unknown>": 0}, {"<char-unknown>": 0}
vocab_id = char_id = 1
print("vocabulary size is ", model.vectors.shape[0])
print("embedding dimension is", model.vectors.shape[1])
model.vectors[0] = [0] * 300
np.save("word_emb.npy", model.vectors)
print("Save Over")
b = np.load("word_emb.npy")
print(b.shape)
print(b[1])

for i, vocab in enumerate(model.vocab):
    if i == 0: continue # jump over "</s>" replace by vocab-unknown
    if vocab in vocab_dict: continue
    vocab_dict[vocab] = vocab_id; vocab_id += 1
    for char in vocab:
        if char not in char_dict:
            char_dict[char] = char_id; char_id += 1
    if i % 1000 == 0:
        print(i, " / 556172   ", vocab_id)

    print(i, " / 556172   ", vocab_id)
with open("vocab.dict", "w") as vf, open("char.dict", "w") as cf:
    vf.write(json.dumps(vocab_dict, ensure_ascii=False) + "\n")
    cf.write(json.dumps(char_dict, ensure_ascii=False) + "\n")
    """for vocab, id in vocab_dict.items():
        vf.write("%s %s t,\n" % (vocab, id))
    for char, id in char_dict.items():
        cf.write("%s %s t,\n" % (char, id))"""

 
