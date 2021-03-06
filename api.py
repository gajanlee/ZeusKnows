"""
The question receptor and answer generator.
"""
import json, threading
import jieba, requests
import numpy as np
import tensorflow as tf

from docs import get_docs
from vote import ensemble_answer
from __init__ import vocabulary, logger

def realtime_process(question, docs):
    datas = []
    question_w_ids = [vocabulary.getVocabID(voca) for voca in list(jieba.cut(question))]
    question_c_ids = [[vocabulary.getCharID(char) for char in voca] for voca in list(jieba.cut(question))]
    for i, doc in enumerate(docs):
        if i == Params.batch_size: break    # the count is satisfied batch size
        passage = doc.passage[:Params.max_p_len]
        tokens = list(jieba.cut(passage))
        datas.append({
            "segmented_question": question_w_ids,
            "segmented_paragraph": [vocabulary.getVocabID(voca) for voca in tokens],
            "char_question": question_c_ids,
            "char_paragraph": [[vocabulary.getCharID(char) for char in voca] for voca in tokens],
            "question_id": 0,   # Later, will transfer from client
            "passage_id": i,
        })
    while len(datas) < Params.batch_size:
        datas.append({
            "segmented_question": [],
            "segmented_paragraph": [],
            "char_question": [[]],
            "char_paragraph": [[]],
            "question_id": 0,   # Later, will transfer from client
            "passage_id": -1,
        })        
    return ljz_load_data(datas, file=False)

from R_net.model import Model
from R_net.params import Params, Vocabulary
from R_net.data_load import ljz_load_data
class R_net_Answer:
    def __init__(self, model):
        with model.graph.as_default():
            sv = tf.train.Supervisor()
            sess = tf.Session()
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
            
            self._dict = Vocabulary()
            self.model = model
            self.sess  = sess
        
    def get_answer(self, question, question_id, query_docs):
        """
        :question :A string of question.
        :query_docs :A list of Document instances.
        """
        data, _ = realtime_process(question, query_docs)
        fd = {m:d for i,(m,d) in enumerate(zip(self.model.data, data))}
        ids, pids = self.sess.run([self.model.output_index, self.model.ids], feed_dict = fd)
        res = []
        for i, (id, doc, pid) in enumerate(zip(ids, query_docs, pids)):
            print(pids)
            if pid[1] == -1: continue
            if id[0] == id[1]: id[1] += 1
            res.append({
                "question_id": question_id,
                "answers": "".join(list(jieba.cut(doc.passage))[id[0]:id[1]]),
                "passage_id": i,
            })
        return res

r_net_answer = R_net_Answer(Model(is_training = False, demo = True))

import os, logging
import pickle
from DuReader.tensorflow.dataset import BRCDataset
from DuReader.tensorflow.rc_model import RCModel
from DuReader.tensorflow.run import build_graph, parse_args

class BIDAF_Answer:
    def __init__(self):
        self.args = parse_args()
        self.rc_model, self.vocab = build_graph(self.args)
        
    def get_answer(self, question, question_id, docs):
        data = [{
            "documents": [{ "segmented_paragraphs": [list(jieba.cut(doc.passage))]}],
            "segmented_question": list(jieba.cut(question)),
            "question_id": question_id,
            "passage_id": i,
        } for i, doc in enumerate(docs)]
        brc_data = BRCDataset(self.args.max_p_num, self.args.max_p_len, self.args.max_q_len,
                          test_files=[data])
        brc_data.convert_to_ids(self.vocab)
        test_batches = brc_data.gen_mini_batches('test', self.args.batch_size,
                                             pad_id=self.vocab.get_id(self.vocab.pad_token), shuffle=False)
        return self.rc_model.evaluate(test_batches, save=False)

bidaf_answer = BIDAF_Answer()

import bottle
from bottle import route, run, request
app = bottle.Bottle()

@app.get("/answer")
def regeiste_answer():
    """
    Test Server status and start construct answer.
    Required:
        1. question
        2. question ID
    """
    print(request.query.question)
    t = threading.Thread(target=api_ensemble, args=(request.query.question, request.query.question_id))
    t.start()
    return {"msg": "ok"}

def api_ensemble(question, question_id):
    docs = get_docs(question)
    anss = bidaf_answer.get_answer(question, question_id, docs) + r_net_answer.get_answer(question, question_id, docs)
    answer = ensemble_answer(question, docs, *anss)
    requests.post("http://0.0.0.0:8080/api/qa/answer/"+question_id, data=json.dumps({"answer": answer[0][1] if answer != "No Answer" else answer, "scores": [ans[0] for ans in answer] if answer != "No Answer" else [],  "answers": [ans[1] for ans in answer] if answer != "No Answer" else [],"passages": [ans[2] for ans in answer] if answer != "No Answer" else []}))
    

if __name__ == "__main__":

    #print("Built model")
    #print(Params.logdir)
    #app.run(port=8081, host='0.0.0.0')
    
    #print(r_net_answer.get_answer("浦发银行电话号码", "23-5", get_docs("浦发银行电话号码")))
    #demo_run = Demo(model)
    #bidaf_answer.get_answer("浦发银行电话号码", get_docs("浦发银行电话号码"))
    #question = "浦发银行电话号码"
    #docs = get_docs(question)
    #anss = r_net_answer.get_answer(question, "235-a", docs)
    #print(ensemble_answer(question, *[ans["answers"] for ans in anss]))
    app.run(port=8081, host='0.0.0.0')