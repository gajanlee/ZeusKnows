"""
The question receptor and answer generator.
"""
import bottle, requests
from bottle import route, run, request
import jieba
import numpy as np
import tensorflow as tf

from R_net.model import Model
from R_net.params import Params, Vocabulary
from R_net.data_load import ljz_load_data

from docs import get_docs
from __init__ import vocabulary

def realtime_process(question, docs):
    datas = []
    question_w_ids = [vocabulary.getVocabID(voca) for voca in list(jieba.cut(question))]
    question_c_ids = [[vocabulary.getCharID(char) for char in voca] for voca in list(jieba.cut(question))]
    for i, doc in enumerate(docs):
        if i == Params.batch_size: break    # the count is satisfied batch size
        passage = doc.passage
        tokens = list(jieba.cut(passage))
        print(doc.passage)        
        datas.append({
            "segmented_question": question_w_ids,
            "segmented_paragraph": [vocabulary.getVocabID(voca) for voca in tokens],
            "char_question": question_c_ids,
            "char_paragraph": [[vocabulary.getCharID(char) for char in voca] for voca in tokens],
            "question_id": 0,   # Later, will transfer from client
            "passage_id": i,
        })
    return ljz_load_data(datas[:1], file=False)


class R_net_Answer:
    def __init__(self, model):
        
        with model.graph.as_default():
            sv = tf.train.Supervisor()
            sess = tf.Session()
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
            
            self._dict = Vocabulary()
            self.model = model
            self.sess  = sess
            
        """
        model.graph.as_default()
        sv = tf.train.Supervisor()
        sess = sv.managed_session()
        sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))"""
        

    
    def get_answer(self, question):
        query_docs = get_docs(question)
        data, shapes = realtime_process(question, query_docs)
        fd = {m:d for i,(m,d) in enumerate(zip(self.model.data, data))}
        ids = self.sess.run([self.model.output_index], feed_dict = fd)
        print(ids)
        ids = ids[0][0]
        if ids[0] == ids[1]:
            ids[1] += 1
        return list(jieba.cut(query_docs[0].passage))[ids[0]:ids[1]]
        #post_answer()
        return response


r_net_answer = R_net_Answer(Model(is_training = False, demo = True))

app = bottle.Bottle()
@app.get('/answer')
def answer():
    question, question_id = request.question, request.question_id
    response = r_net_answer.get_answer(question)
    return {"answer": "".join(response), "passages": []}    
    #global query_question, query_docs, response, question_id
    #query_question, query_docs, question_id = request.query.question, get_docs(request.query.question), request.query.question_id
    return {"msg": "ok"}


if __name__ == "__main__":

    print("Built model")
    print(Params.logdir)
    app.run(port=8081, host='0.0.0.0')
    
    r_net_answer.get_answer("浦发银行电话号码")
    #demo_run = Demo(model)