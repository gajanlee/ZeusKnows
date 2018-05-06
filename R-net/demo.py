#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function

import tensorflow as tf
import bottle
from bottle import route, run, request
import threading

from params import Params
#from process import *
from time import sleep
from params import Vocabulary
from ZeusKnows.docs import get_docs
from ZeusKnows import vocabulary

app = bottle.Bottle()
query_question, query_docs = None, None
response = ""

@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html

@app.get('/answer')
def answer():
    global query_question, query_docs
    query_question, query_docs = request.query.question, get_docs(request.query.question)

    while not response:
        sleep(0.1)
    print("received response: {}".format(response))
    Final_response = {"answer": response}
    response = []
    return Final_response

class Demo(object):
    def __init__(self, model):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args = [model, run_event]).start()
        app.run(port=8080, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    def demo_backend(self, model, run_event):
        global query_question, query_docs, response
        #dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
        dict_ = Vocabulary()

        with model.graph.as_default():
            sv = tf.train.Supervisor()
            with sv.managed_session() as sess:
                sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
                while run_event.is_set():
                    sleep(0.1)
                    if query_question:
                        data, shapes = realtime_process(query_question, query_docs)
                        fd = {m:d for i,(m,d) in enumerate(zip(model.data, data))}
                        ids = sess.run([model.output_index], feed_dict = fd)
                        
                        print(ids)
                        ids = ids[0][0]
                        if ids[0] == ids[1]:
                            ids[1] += 1
                        passage_t = tokenize_corenlp(query[0])
                        response = " ".join(passage_t[ids[0]:ids[1]])
                        query_question = ""     # clear the question

import jieba
import numpy as np
from data_load import ljz_load_data
def realtime_process(question, docs):
    datas = []
    question_w_ids = [vocabulary.getVocabID(voca) for voca in list(jieba.cut(question))]
    question_c_ids = [vocabulary.getCharID(char) for char in question]
    for i, doc in enumerate(docs):
        if i == Params.batch_size: break    # the count is satisfied batch size
        passage = doc.passage
        tokens = list(jieba.cut(passage))        
        datas.append({
            "segmented_question": question_w_ids,
            "segmented_paragraph": [vocabulary.getVocabID(voca) for voca in tokens],
            "char_question": question_c_ids,
            "char_passage": [vocabulary.getCharID(char) for char in passage],
        })
    return ljz_load_data(datas, file=False)

if __name__ == "__main__":
    #app.run(port=8081, host='0.0.0.0')
    realtime_process("", "")