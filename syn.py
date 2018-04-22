from utils import *
import json, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="running mode: rerank, entity", default="rerank", type=str)


parser.add_argument("-v", "--verbose", help="output verbose information", action="store_true")
parser.add_argument("-d", "--debug", help="print more information in debug mode", action="store_true")
parser.add_argument("--max_p", help="load max passage len", default=450, type=int)
parser.add_argument("--r_net", help="r_net result file path", default="./res/search.res", type=str)
parser.add_argument("--bidaf", help="bidaf result file paths, a list of items", nargs="*", default="../../09result.json", type=str)

parser.add_argument("--input", help="unentity answer file", default="unentity.json") # if mode is entity answer, it is necessary
parser.add_argument("--output", help="synthetic result output file path", default=str(time.localtime(time.time()).tm_mday)+"result.json")   # as 20result.json
args = parser.parse_args()


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
        self.load_test_result(args.r_net)
        self.load_bidaf_result(args.bidaf)

    @logging_util
    def load_test_data(self, path):
        """
            Load Data, testing data
        """
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if args.verbose and i % 10000 == 0: logger.info("Now Line %s" % i)
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
                ans = expand_answer(self.lookup[d["question_id"]]["passages"][d["passage_id"]], d["spans"])
                # ans = self.lookup[d["question_id"]]["passages"][d["passage_id"]][d["spans"][0]: d["spans"][1]]  # We need expand up/down spans.
                if d["question_id"] not in self.r_net_result:
                    self.r_net_result[d["question_id"]] = {
                        "question_id": d["question_id"],
                        "question_type": self.lookup[d["question_id"]]["question_type"],
                        "question": self.lookup[d["question_id"]]["segmented_q"],
                        "answers": [ans], 
                        "yesno_answers": [],
                        "entity_answers": [[]],
                    }
                else: 
                    self.r_net_result[d["question_id"]]["answers"].append(ans)
                if args.verbose and i % 10000 == 0: logger.info("Loaded R-net Result, Line now is %s" % i)
                if args.debug and i == 200: logger.info("Debug Model, Loaded R-net Data Done"); break

        
    @logging_util
    def load_bidaf_result(self, paths):
        if type(paths) is str: paths = [paths]
        assert type(paths) is list
        for path in paths:
            with open(path) as r:
                logger.info("Loading BIDAF Result, path is %s" % path)
                if not hasattr(self, "bidaf_result"): self.bidaf_result = { json.loads(line)["question_id"]: json.loads(line) for line in r }
                else: [self.bidaf_result[json.loads(line)["question_id"]]["answers"].extend(json.loads(line)["answers"]) for line in r]

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
    with open(args.output, "w") as w:
        [w.write(json.dumps(d, ensure_ascii=False) + "\n") for d in result.values()]


def _base_valid(ans):
    return ans not in ["。", "", None]
    

stat = [0, 0, 0]
def _get_best(*ans_jsons):
    """
    ans_jsons is a set of candidate data
    """
    anses = []
    for i, ans_json in enumerate(ans_jsons):
        anses.extend([(replace_invisible(ans), i) for ans in ans_json["answers"] if _base_valid(ans)])
        #anses = [ans for ans in ans_json["answers"] if _base_valid(ans)]
        # anses.extend(ans_json["answers"])
    if args.debug:
        print(anses)
    scores = []
    for i, (a1, _bl) in enumerate(anses):
        scr = 0
        for j, (a2, _) in enumerate(anses):
            if i == j: continue
            scr += score(a1, a2)
        scores.append((scr, a1, _bl))
    scores.sort(key=lambda x: x[0], reverse=True)
    if args.debug:
        print(scores, "\n")
    
    del ans_jsons[0]["question"]
    if len(scores) == 0: print(ans_jsons); ans_jsons[0]["answers"] = ans_jsons[1]["answers"][0]; stat[1] += 1
    else: ans_jsons[0]["answers"] = [scores[0][1]]; """[s[1] for s in scores[:2]]"""; stat[scores[0][2]] += 1
    # ans_jsons[0]["entity_answers"] = entity(ans_jsons[0]["answers"][0]), pypy3 doesn't have jieba module
    return ans_jsons[0]
    
        

def rerank(result):
    """
    TODO:
        get all result file ( if ensemble, contains all answer)
        assign final answer to result
    """
    res = {}
    for i, (q_id, ans) in enumerate(result.items(), 1):
        res[q_id] = _get_best(ans, data_handler.get_bidaf_by_id(q_id))
        if args.verbose and i % 1000 == 0: logger.info("Reranking count %s / 60000" % i)

    for q_id, d in data_handler.get_all_bidaf().items():
        d["answers"] = [d["answers"][0]]
        if q_id not in res: res[q_id] = d
    logger.info("R-net / Bidaf ==> %s / %s" % (i, 60000-i))
    write(res)


if __name__ == "__main__":
    if args.mode == "rerank":
        rerank(data_handler.r_net_result)
        logger.info("Selected: R-net / Bidaf ==> %s / %s" % (stat[0], stat[1]))
    elif args.mode == "entity":
        with open(args.input) as input:
            res = []
            for line in input:
                d = json.loads(line)
                d["entity_answers"] = entity(d["answers"][0]) if d["question_type"] == "ENTITY" else [[]]
                res.append(d)
        with open(args.output, "w") as output:
            output.write("\n".join([json.dumps(d) for d in res]))
