from __init__ import *
import json
import jieba
from bs4 import BeautifulSoup
from multiprocessing import Process, Lock, Pool, TimeoutError
struct_file = "./upload_res.json"
output_file = "./result.json"
#from bleu_metric.bleu  import Bleu
from rouge_metric.rouge import Rouge
#Bleu().compute_score(a, [b])[0][3]

# lookup = Lookup()

idf = json.load(open("idf.stat"))
article_count = 1294232

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
    if not (len(question) and len(passage)):
        return 0
    if not (vocabulary.getVocabID(question) and vocabulary.getVocabID(passage)):
        #print(Bleu(1).compute_score(question, passage), question, passage)
        #cos = find / len(p)
        cos = Rouge().calc_score([passage], [question]) * 2 - 1
    else:
        cos = cosine(model.get_vector(question), model.get_vector(passage))
    
    return math.exp(cos**2 if cos >= 0 else -cos**2)

from itertools import product

def score(question, passage, tf_idf):
    """
        question: a token of question, string
        passage : a token too. string
        tf_idf: passage token's tf_idf
    """
    return get_cos(question, passage) * tf_idf


def match_score_p(question, passage, r):
    """
        question: a list of segmented question
        passage : list, segmented
    """
    count = {}
    for p in passage:
        count[p] = count.get(p, 0) + 1
    scores = list(filter(lambda x: x != 0, [score(q, p, count.get(q, 1) * math.log(article_count / idf.get(p, 1))) for q, p in product(question, passage)]))  # filter 0

    return sum(scores) / len(scores) * (1/(r+1))


writerT = open("search/total_test.stat", "w")

def Write(data):
    writerT.write(json.dumps(data, ensure_ascii=False) + "\n")

def Close():
    writerT.close()


from collections import Counter
def _match(can, ref):
    """
        candidate is para / title
        ref is question / question
    """
    return sum((Counter(can) & Counter(ref)).values()) / len(ref)

def match_score(question, para, title):
    """
        question and title's match_score as weight
        question and para's match_score * weigth as return score
    """
    weight = _match(title, question)
    return weight * _match(question, para)


def process():
    passage_id = 0
    with open("./search.test.json3") as f:
        for i, line in enumerate(f, 1):
            
            if i % 10 == 0: logger.info("%s / %s" % (i, 30000))
            #print(i)
            #if i == 10: break
            line = json.loads(line)
            output = []
            for doc in line["documents"]:
                for para in doc["segmented_paragraphs"]:
                    
                    if len(para) == 0: continue
                    if para[0] == "<":
                        para = list(jieba.cut(BeautifulSoup("".join(para), "html.parser").text))
                    if len(para) == 0: continue
                    if len(para) > 500: continue
                    output.append((match_score(line["segmented_question"], para, doc["title"]), para))

            output.sort(key=lambda x: x[0], reverse=True)
            #sorted(output, key=lambda x: x[0])
            print(output)
            for o in output[:3]:
                d = {
                    "question_id": line["question_id"],
                    "question_type": line["question_type"],
                    "passage_id": passage_id,
                    "segmented_paragraph": [vocabulary.getVocabID(v) for v in o[1]],
                    "char_paragraph": [[vocabulary.getCharID(c) for c in v] for v in o[1]],
                    "segmented_question": [vocabulary.getVocabID(v) for v in line["segmented_question"]],
                    "char_question": [[vocabulary.getCharID(c) for c in v] for v in line["segmented_question"]],
                    "score": o[0],
                }
                #Write_ID(d)
                d["segmented_p"] = o[1]
                d["segmented_q"] = line["segmented_question"]
                Write(d)
                passage_id += 1
    Close()


if __name__ == "__main__":
    for q, p, t in  gen_test():
        print(match_score(q, p ,t), p)

def gen_test():
    for doc in test()["documents"]:
        for para in doc["segmented_paragraphs"]:
            yield test()["segmented_question"], para, doc["title"]

def test():
    return json.load(open("test.case"))