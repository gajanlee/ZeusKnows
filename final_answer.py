from __init__ import *
#import json
struct_file = "./upload_res.json"
output_file = "./result.json"

lookup = Lookup()

idf = json.load(open("idf.stat"))

class DataHandler:
    datas = {}
    def __init__(self):
        self.get_data()

    def get_data(self):
        with open(Params.struct_file) as f:
            for line in f:
                d = json.loads(line)
                if d["question_id"] in self.datas:
                    d["spans"].append(d["spans"])
                    d["answers"].append(lookup.get_answer_content_by_passageID_spans(d["passage_id"], d["spans"]))
                    if len(d["answers"]) >= 3:
                        print(d["question_id"])
                else:
                    self.datas[d["question_id"]] = {
                        "question_id": d["question_id"],
                        "question_type": lookup.get_question_type_by_questionID(d["question_id"]),
                        "answers": [lookup.get_answer_content_by_passageID_spans(d["passage_id"], d["spans"])],
                        "yesno_answers": [],
                        "entity_answers": [[]],
                        "spans": [d["spans"]], 
                    }
    
    def process_data(self):

        for answer in self.datas:
            if answer["question_type"] == "YES_NO":
                pass
            elif answer["question_type"] == "YES_NO":
                pass


    def align_data(self):
        pass

import word2vec, math
model = word2vec.load("wordsVec.bin")


def cosine(vec1, vec2):
    """
        vec1, vec2: two vector, no limited dimesi
    """
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for x1, x2 in zip(vec1, vec2):  
        dot_product += x1 * x2  
        normA += x1 ** 2  
        normB += x2 ** 2  
    """if normA == 0.0 or normB==0.0:  
        return None
    else:  """
    return dot_product / ((normA*normB)**0.5)  

def get_cos(question, passage):
    if not (vocabulary.getVocabID(question) and vocabulary.getVocabID(passage)):
        score = 0
        for char_pair in product(question, passage):
            score += (1 if char_pair[0] == char_pair[1] else -1)
        score /= len(char_pair)
    else:
        cos = cosine(model.get_vector(question), model.get_vector(passage))
    
    return (math.exp(-cos) if cos >= 0 else math.exp(cos))

from itertools import product

def score(question, passage, tf_idf):
    """
        question: a token of question, string
        passage : a token too. string
        tf_idf: passage token's tf_idf
    """
    return get_cos(question, passage) * tf_idf


def match_score(question, passage):
    """
        question: a list of segmented question
        passage : list, segmented
    """
    count = {}
    for p in passage:
        count[p] = count.get(p) + 1
    scores = filter(lambda x: x != 0, [score(*pair, count.get(pair[0], 1) * idf.get(pair[1], math.log(article_count))) for pair in product(question, passage)])  # filter 0

    return sum(scores) / len(scores)



if __name__ == "__main__":
    #DataHandler().process_data()  
    """res = {}
    with open("./result.json") as f:
        for line in f:
            d = json.loads(line)
            if d["question_id"] in res:
                res[d["question_id"]]["answers"].append(d["answers"][0])
            else:
                res[d["question_id"]] = d

    with open("result2.json", "w") as f:
        for r in res.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")"""
