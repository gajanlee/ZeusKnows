# -*- coding: utf-8 -*-
#/usr/bin/python2

from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

from params import Params
import numpy as np
import tensorflow as tf
#from process import *
#from sklearn.model_selection import train_test_split

# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.
    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

def load_data(dir_):
    # Target indices
    indices = load_target(dir_ + Params.target_dir)

    # Load question data
    print("Loading question data...")
    q_word_ids, _ = load_word(dir_ + Params.q_word_dir)
    q_char_ids, q_char_len, q_word_len = load_char(dir_ + Params.q_chars_dir)

    # Load passage data
    print("Loading passage data...")
    p_word_ids, _ = load_word(dir_ + Params.p_word_dir)
    p_char_ids, p_char_len, p_word_len = load_char(dir_ + Params.p_chars_dir)

    # Get max length to pad
    p_max_word = Params.max_p_len#np.max(p_word_len)
    p_max_char = Params.max_char_len#,max_value(p_char_len))
    q_max_word = Params.max_q_len#,np.max(q_word_len)
    q_max_char = Params.max_char_len#,max_value(q_char_len))

    # pad_data
    print("Preparing data...")
    p_word_ids = pad_data(p_word_ids,p_max_word)
    q_word_ids = pad_data(q_word_ids,q_max_word)
    p_char_ids = pad_char_data(p_char_ids,p_max_char,p_max_word)
    q_char_ids = pad_char_data(q_char_ids,q_max_char,q_max_word)

    # to numpy
    indices = np.reshape(np.asarray(indices,np.int32),(-1,2))
    p_word_len = np.reshape(np.asarray(p_word_len,np.int32),(-1,1))
    q_word_len = np.reshape(np.asarray(q_word_len,np.int32),(-1,1))
    # p_char_len = pad_data(p_char_len,p_max_word)
    # q_char_len = pad_data(q_char_len,q_max_word)
    p_char_len = pad_char_len(p_char_len, p_max_word, p_max_char)
    q_char_len = pad_char_len(q_char_len, q_max_word, q_max_char)

    for i in range(p_word_len.shape[0]):
        if p_word_len[i,0] > p_max_word:
            p_word_len[i,0] = p_max_word
    for i in range(q_word_len.shape[0]):
        if q_word_len[i,0] > q_max_word:
            q_word_len[i,0] = q_max_word

    # shapes of each data
    shapes=[(p_max_word,),(q_max_word,),
            (p_max_word,p_max_char,),(q_max_word,q_max_char,),
            (1,),(1,),
            (p_max_word,),(q_max_word,),
            (2,)]

    return ([p_word_ids, q_word_ids,
            p_char_ids, q_char_ids,
            p_word_len, q_word_len,
            p_char_len, q_char_len,
            indices], shapes)

def get_dev():
    print("start load dev set...")
    devset, shapes = ljz_load_data(Params.dev_dir)  #Params.dev_dir)
    #devset, shapes = ljz_load_data(Params.test_rank_dir)
    indices = devset[-3]
    # devset = [np.reshape(input_, shapes[i]) for i,input_ in enumerate(devset)]

    dev_ind = np.arange(indices.shape[0],dtype = np.int32)
    np.random.shuffle(dev_ind)
    print("dev ok!")
    return devset, dev_ind

def get_batch(is_training = True):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load dataset
        if Params.mode.lower() == "gen_rank":
            input_list, shapes = ljz_load_data(Params.test_rank_dir)
        else:
            input_list, shapes = ljz_load_data(Params.train_dir if is_training else Params.dev_dir)
        #for s in input_list:
            #print(s.shape)
        print(len(input_list))
        #np.reshape(input_list[2][2], (500, 8))
        indices = input_list[-3]    # Warning.

        train_ind = np.arange(indices.shape[0],dtype = np.int32)
        np.random.shuffle(train_ind)
        print(train_ind)
        size = Params.data_size
        if Params.data_size > indices.shape[0] or Params.data_size == -1:
            size = indices.shape[0]
        ind_list = tf.convert_to_tensor(train_ind[:size])
        print(size)

        # Create Queues
        ind_list = tf.train.slice_input_producer([ind_list], shuffle=True)

        @producer_func
        def get_data(ind):
            '''From `_inputs`, which has been fetched from slice queues,
               then enqueue them again.
            '''
            return [np.reshape(input_[ind], shapes[i]) for i,input_ in enumerate(input_list)]

        data = get_data(inputs=ind_list,
                        dtypes=[np.int32]*11,
                        capacity=Params.batch_size*32,
                        num_threads=6)
        #[np.reshape(input_[ind], shapes[i] for i, input_ in enumerate(input_list))
        # create batch queues
        batch = tf.train.batch(data,
                                shapes=shapes,
                                num_threads=2,
                                batch_size=Params.batch_size,
                                capacity=Params.batch_size*32,
                                dynamic_pad=True)
    for i in input_list:
        print(len(i))
    print("train ok!")
    print(batch)
    return batch, size // Params.batch_size

from copy import deepcopy as copy
def padding_data(data, max_len):
    t = copy(data)
    t.extend([0] * (max_len - len(data)))
    return t


def padding_char_data(data, max_len, max_clen, ith=None):
    t = copy(data)
    for k, p in enumerate(t):
        if len(p) > max_clen: t[k] = [0]*max_clen
        p.extend([0] * (max_clen - len(p)))
    t.extend( [[0] * max_clen] * (max_len - len(t)))
    return t


def padding_char_len(data, max_len):
    t = copy(data)
    for p in t:
        p.extend([0] * (max_len - len(p)))
    return t

import json, numpy as np
def ljz_load_data(_file, file=True):
    #_file = "../../Zeus/description_id.stat"  # description
    passage_word_ids, question_word_ids = [], []
    passage_char_ids, question_char_ids = [], []
    passage_word_len, question_word_len = [], []
    passage_char_len, question_char_len = [], []
    indices = []
    tags = []
    ids = []    # question's id

    max_plen, max_qlen, max_clen = Params.max_p_len, Params.max_q_len, Params.max_char_len
    print("loading data dir is ", _file)

    if file == True:
        fp = open(_file)
    else:
        fp = _file
    for i, line in enumerate(fp):
        #if i == 2000: break
        #print(i)
        if i % 1000 == 0:
            print("data loading %s line" % i)
        if file == True:
            d = json.loads(line)
        else:
            d = line
        if len(d["segmented_paragraph"]) > max_plen or len(d["segmented_question"]) > max_qlen:
            #print(len(d["segmented_paragraph"]), len(d["segmented_question"]))
            continue
        passage_word_ids.append( padding_data(d["segmented_paragraph"], max_plen))
        question_word_ids.append( padding_data(d["segmented_question"], max_qlen))
        passage_char_ids.append( padding_char_data(d["char_paragraph"], max_plen, max_clen, i))
        question_char_ids.append( padding_char_data(d["char_question"], max_qlen, max_clen))

        passage_word_len.append([len(d["segmented_paragraph"])])
        question_word_len.append([len(d["segmented_question"])])
        passage_char_len.append([min(len(word), 8) for word in d["char_paragraph"]])
        question_char_len.append([min(len(word), 8) for word in d["char_question"]])
        if Params.mode.lower() == "prank":
            indices.append([0, 0])
            tags.append(d["tag"])
        elif Params.mode.lower() in ["gen_rank", "test", "demo"]:
            indices.append([0, 0])
            tags.append([0])
            ids.append([d["question_id"], d["passage_id"]])
        elif Params.mode == "dev":
            indices.append([d["answer_spans"][0][0], d["answer_spans"][0][1]+1])
            tags.append([0])
            ids.append()
        else:
            indices.append([d["answer_spans"][0][0], d["answer_spans"][0][1]+1])
            tags.append([0])
            ids.append([0,0])
        #ids.append([d["question_id"], d["passage_id"]])
        #ids.append([0, 0])
        # to numpy
        
        
        indices = np.reshape(np.asarray(indices,np.int32),(-1,2))
        tags = np.reshape(np.asarray(tags, np.float32), (-1, 1))
        ids = np.reshape(np.asarray(ids, np.int32), (-1, 2))
        passage_word_len = np.reshape(np.asarray(passage_word_len, np.int32),(-1,1))
        question_word_len = np.reshape(np.asarray(question_word_len, np.int32),(-1,1))
        # p_char_len = pad_data(p_char_len,p_max_word)
        # q_char_len = pad_data(q_char_len,q_max_word)
        p_char_len = padding_char_len(passage_char_len, max_plen)
        q_char_len = padding_char_len(question_char_len, max_qlen)
        # shapes of each data
        shapes=[(max_plen,),(max_qlen,),
                (max_plen, max_clen,),(max_qlen,max_clen,),
                (1,),(1,),
                (max_plen,),(max_qlen,),
                (2,), (1,), (2,)]
       
        return ([np.array(passage_word_ids), np.array(question_word_ids),
                np.array(passage_char_ids), np.array(question_char_ids),
                passage_word_len, question_word_len,
                np.array(p_char_len), np.array(q_char_len),
                indices, tags, ids], shapes)

if __name__ == "__main__":
    a, b = get_batch()
    #print(a[6], a[7], a[8], b) 
    
