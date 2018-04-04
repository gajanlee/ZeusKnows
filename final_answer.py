from __init__ import *
#import json
struct_file = "./upload_res.json"
output_file = "./result.json"


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

def match_score(question, passage):
    

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
