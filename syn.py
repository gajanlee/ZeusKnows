from utils import *
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="output verbose information", action="store_true")
parser.add_argument("-d", "--debug", help="print more information in debug mode", action="store_true")
parser.add_argument("--max_p", help="load max passage len", default=450, type=int)
args = parser.parse_args()
print(args.verbose, args.debug)


class Config:
    ensemble = False

    @staticmethod
    def required_ensemble(func):
        def check():
            assert Config.ensemble == True
            return func()
        return check

class DataHandler:
    """
    Contains All testing data, 
    """
    def __init__(self):
        self.lookup = {}    # It is used to find passage and question for "R-net" model
        self.r_net_result = {}     # It stores R-net model result (after synthesis).

        self.load_test_data("./total.test.net.json")
        self.load_test_result("./14_total_total_model.res")
        self.load_bidaf_result("../../09result.json")

    @logging_util
    def load_test_data(self, path):
        """
            Load Data, testing data
        """
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if args.verbose and i % 1000 == 0: logger.info("Now Line %s" % i)
                data = json.loads(line)
                if len(data["segmented_p"]) > args.max_p: continue
                if data["question_id"] not in self.lookup:
                    del data["segmented_question"], data["segmented_paragraph"]
                    del data["char_question"], data["char_paragraph"]
                    p_id, p = data["passage_id"], data["segmented_p"]
                    del data["segmented_p"], data["passage_id"]
                    data["passages"] = { p_id: p}
                    self.lookup[data["question_id"]] = data
                else:
                    self.lookup[data["question_id"]]["passages"][data["passage_id"]] = data["segmented_p"]
    
    @logging_util
    def load_test_result(self, path):
        """
        Load R_net Test Result, Recover R-net output to origin json.
        :path: R-net output json file path.

        # Example
        path file contains R-net result;
        {"passage_id": 1, "question_id": 0, answer_spans: [0, 1]}

        :construct self.r_net_result
        {"question_id": 0, "answer": [token1, token2, ...], "question": [token1, token2, ...]}
        """
        with open(path) as f:
            for i, line in enumerate(f, 1):
                d = json.loads(line)
                ans = self.lookup[d["question_id"]]["passages"][d["passage_id"]][d["spans"][0]: d["spans"][1]]
                if d["question_id"] not in self.r_net_result:
                    self.r_net_result[d["question_id"]] = {
                        "question_id": d["question_id"],
                        "question_type": self.lookup[d["question_id"]]["question_type"],
                        "question": self.lookup[d["question_id"]]["segmented_q"],
                        "answers": ["".join(ans)],
                        "yesno_answers": [],
                        "entity_answers": [[]],
                    }
                else: 
                    self.r_net_result[d["question_id"]]["answers"].append("".join(ans))
                if args.verbose and i % 1000 == 0: logger.info("Loaded R-net Result, Line now is %s" % i)
                if args.debug and i == 200: logger.info("Debug Model, Loaded R-net Data Done"); break

        
    @logging_util
    def load_bidaf_result(self, path):
        with open(path) as r:
            self.bidaf_result = { json.loads(line)["question_id"]: json.loads(line) for line in r }

    #@Config.required_ensemble
    def get_all_bidaf(self):
        return self.bidaf_result

    def get_bidaf_by_id(self, id):
        return self.bidaf_result[id]
    
data_handler = DataHandler()


def write(result):
    """
    TODO:
        Write Final Answers to a File.
    """
    with open("15r_net_total.json", "w") as w:
        [w.write(json.dumps(d, ensure_ascii=False) + "\n") for d in result.values()]


def _get_best(*ans_jsons):
    """
    ans_jsons is a set of candidate data
    """
    anses = []
    for ans_json in ans_jsons:
        [anses.append(ans) for ans in ans_json["answers"] if len(ans) != 0]
        # anses.extend(ans_json["answers"])

    scores = []
    for i, a1 in enumerate(anses):
        scr = 0
        for j, a2 in enumerate(anses):
            if i == j: continue
            scr += score(a1, a2)
        scores.append((scr, a1))
    scores.sort(key=lambda x: x[0], reverse=True)
    if args.debug:
        print(scores, "\n")
    
    del ans_jsons[0]["question"]
    ans_jsons[0]["answers"] = [scores[0][1]]
    return ans_jsons[0]
    
        

def rerank(result):
    """
    TODO:
        get all result file ( if ensemble, contains all answer)
        assign final answer to result
    """
    res = {}
    for i, (q_id, ans) in enumerate(result.items()):
        res[q_id] = _get_best(ans, data_handler.get_bidaf_by_id(q_id))
        if args.verbose and i % 1000 == 0: logger.info("Reranking Line %s" % i)

    for q_id, d in data_handler.get_all_bidaf().items():
        if q_id not in res: res[q_id] = d
    logger.info("R-net / Bidaf ==> %s / %s" % (i, 60000-i))
    write(res)


if __name__ == "__main__":
    rerank(data_handler.r_net_result)
    


