import json



class Config:
    ensemble = False

    @staticmethod
    def required_ensemble(func):
        def check():
            assert Config.ensemble == True
            return func()
        return inner

class DataHandler:
    """
    Contains All testing data, 
    """
    def __init__(self, ensemble=False):
        self.load_test_data()
        if ensemble: self.load_bidaf_result()

    def load_test_data(self, path):
        """
            Load Data, testing data
        """
        with open(path) as f:
            self.data = [json.loads(line) for line in f]
    
    def load_bidaf_result(self):
        with open("../../07result.json") as r:
            self.bidaf_result = { json.loads(line)["question_id"]: json.loads(line) for line in f }


    def lookup_by_passage_id(self, passage_id):
        return self.data[passage_id]
                
    #@Config.required_ensemble
    def get_all_bidaf(self):
        return self.bidaf_result
    
data_handler = DataHandler(ensemble=True)

def recover_from_rnet(data_json):
    """
        :data_json: R-net output json.

        # Example
        {"passage_id": 1, "question_id": 0, answer_spans: [0, 1]}

        :return
        {"question_id": 0, "answer": [token1, token2, ...], "question": [token1, token2, ...]}
    """
    d = data_handler.lookup_by_passage_id(data_json["passage_id"])
    return {
        "question_id": d["question_id"],
        "question_type": d["question_type"],
        "answer": d["segmented_p"][d["spans"][0]: d["spans"][1]],
        "question": d["question"]
    }

def get_all_results():
    results = {}
    with open("r_net output") as f:
        for line in f:
            d = recover_from_rnet(json.loads(line))
            if d["question_id"] in results:
                results[d["question_id"]].answers.append( d["answer"])
            else:
                results[d["question_id"]] = {
                    "question_id" : d["question_id"],
                    "question_type" : d["question_type"],
                    "answers" : [d["answer"]],
                    "question" : d["question"],}
    
    if Config.ensemble:
        for question_id, data in data_handler.get_all_bidaf().items():
            if question_id in results: results.extend(data["answers"])   # all data in it
            else: results[question_id] = data
    
    return results

def rerank(result):
    """
    TODO:
        get all result file ( if ensemble, contains all answer)
        assign final answer to result
    """
    pass

def write():
    """
    TODO:
        Write Final Answers to a File.
    """
    pass

if __name__ == "__main__":
    
    


