from __init__ import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tp", default="train", help="the type of generated data type")
args = parser.parse_args()

class Process:
    def __init__(self, tp):
        self.writers = {
            "SEARCH": open("search.{}.net.json".format(tp), "w"),
            "ZHIDAO": open("zhidao.{}.net.json".format(tp), "w"),
            "TOTAL": open("total.{}.net.json".format(tp), "w"),
        }
        self.start(tp)
        self.close()
        self.passage_id = 0

    def start(self, tp):
        for _file, _mode in zip(["../DuReader/{tp}set/search.{tp}.json".format(tp=tp), "../DuReader/{tp}set/zhidao.{tp}.json".format(tp=tp)], ["SEARCH", "ZHIDAO"]):
            with open(_file) as r:
                for i, line in enumerate(r, 1):
                    if tp == "test": d = self.test_process(json.loads(line))
                    elif tp in ["train", "dev"]: d = self.train_process(json.loads(line))
                    if d is None: continue
                    [self.writers[_m].write(json.dumps(d, ensure_ascii=False) + "\n") for _m in [_mode, "TOTAL"]]
                    if False and i >= 101: break
                    if i % 100 == 0: print(i, tp, _mode)

    def train_process(self, data):
        if not data["match_scores"]: return
        answer_spans = data["answer_spans"][0] # Actually it only ones
        try: doc = data["documents"][data["answer_docs"][0]]
        except: logger.info("error doc"); return

        base_format = {
            "question_type": data["question_type"],    # YES_NO / ENTITY / DESCRIPTION$
            "question_id": data["question_id"],        # ID(int)$
            "fact_or_opinion": data["fact_or_opinion"],# FACT  / OPINION$
            "segmented_p": doc["segmented_paragraphs"][doc["most_related_para"]],  # Now, only one paragraph, shape [para_len]$
            "match_scores": data["match_scores"],
            "answer_spans": data["answer_spans"],
            "segmented_q": data["segmented_question"]}
        
        base_format["char_paragraph"] = [vocabulary.getCharID(word, True) for word in base_format["segmented_p"]]
        base_format["char_question"] = [vocabulary.getCharID(word, True) for word in base_format["segmented_q"]]
        
        base_format["segmented_paragraph"] = [vocabulary.getVocabID(v) for v in base_format["segmented_p"]]
        base_format["segmented_question"] = [vocabulary.getVocabID(v) for v in base_format["segmented_q"]]
        
        return base_format

    def test_process(self, data):
        format = {
            "question_id": data["question_id"],
            "question_type": data["question_type"],
            "segmented_question": data["segmented_question"],
            "char_question": [vocabulary.getCharID(word, True) for word in data["segmented_question"]]
        }

        for doc in data["documents"]:
            pass
    
    def close(self):
        for writer in self.writers.values():
            writer.close()

if __name__ == "__main__":
    Process(args.tp)
