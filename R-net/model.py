# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, get_dev
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
from evaluate import *
import numpy as np
import cPickle as pickle
#from process import *
#from demo import Demo
import os

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
            "adam":tf.train.AdamOptimizer,
            "gradientdescent":tf.train.GradientDescentOptimizer,
            "adagrad":tf.train.AdagradOptimizer}

class Vocabulary:
    def __init__(self):
        self.vocab_dict, self.char_dict = {}, {}
        self.load_vocab_dict("../vocab.dict")
        self.load_char_dict("../char.dict")
        print("vocab_dict size is ", len(self.vocab_dict))
        print("char_dict size is ", len(self.char_dict))

    def load_vocab_dict(self, vocab_path):
        vocab_dict = json.load(open(vocab_path))
        for vocab, _id in vocab_dict.items():
            self.vocab_dict[_id] = vocab
        print("=======>", self.vocab_dict[100])

        #with open(vocab_path) as fp: 
            #for i, line in enumerate(fp):
                #self.vocab_dict[i] = line.split(" ")[0] 
                #self.vocab_dict[res[1]] = int(res[0])

    def load_char_dict(self, char_path):
        char_dict = json.load(open(char_path))
        for char, _id in char_dict.items():
            self.char_dict[_id] = char
        print("======>", self.char_dict[100])
        #with open(char_path) as fp: 
            #for i, line in enumerate(fp, 1): 
                #self.char_dict[line[:-1]] = i 
                #self.char_dict[i] = line.split(" ")[0]
                #res = line.split(' ')
                #print(res[0], '+', res[1])
                #self.char_dict[res[0]] = int(res[1])
                #self.char_dict[int(res[1])] = res[0]

    def ind2word(self,ids):
        output = []
        for i in ids:
            output.append(self.vocab_dict[i])
        return " ".join(output)

    def ind2char(self,ids):
        output = []
        for i in ids:
            for j in i:
                output.append(self.char_dict[j])
            output.append(" ")
        return "".join(output)


class Model(object):
    def __init__(self,is_training = True, demo = False):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if demo:
                self.passage_w = tf.placeholder(tf.int32,
                                        [1, Params.max_p_len,],"passage_w")
                self.question_w = tf.placeholder(tf.int32,
                                        [1, Params.max_q_len,],"passage_q")
                self.passage_c = tf.placeholder(tf.int32,
                                        [1, Params.max_p_len,Params.max_char_len],"passage_pc")
                self.question_c = tf.placeholder(tf.int32,
                                        [1, Params.max_q_len,Params.max_char_len],"passage_qc")
                self.passage_w_len_ = tf.placeholder(tf.int32,
                                        [1,1],"passage_w_len_")
                self.question_w_len_ = tf.placeholder(tf.int32,
                                        [1,1],"question_w_len_")
                self.passage_c_len = tf.placeholder(tf.int32,
                                        [1, Params.max_p_len],"passage_c_len")
                self.question_c_len = tf.placeholder(tf.int32,
                                        [1, Params.max_q_len],"question_c_len")
                self.data = (self.passage_w,
                            self.question_w,
                            self.passage_c,
                            self.question_c,
                            self.passage_w_len_,
                            self.question_w_len_,
                            self.passage_c_len,
                            self.question_c_len)
            else:
                self.data, self.num_batch = get_batch(is_training = is_training)
                (self.passage_w,
                self.question_w,
                self.passage_c,
                self.question_c,
                self.passage_w_len_,
                self.question_w_len_,
                self.passage_c_len,
                self.question_c_len,
                self.indices,
                self.tags,
                self.ids ) = self.data
                
            self.passage_w_len = tf.squeeze(self.passage_w_len_, -1)
            self.question_w_len = tf.squeeze(self.question_w_len_, -1)
            
            #with tf.device("/gpu:0"):
            self.encode_ids()
            self.params = get_attn_params(Params.attn_size, initializer = tf.contrib.layers.xavier_initializer)
            #with tf.device("/gpu:1"):
            self.attention_match_rnn()
            self.bidirectional_readout()
            self.pointer_network()
            self.outputs()
            
            #self.passage_rank()
            if is_training:
                #self.p_rank_loss()
                self.loss_function()
                self.summary()
                self.init_op = tf.global_variables_initializer()
            total_params()

    def encode_ids(self):
        with tf.device('/cpu:0'):
            self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.char_vocab_size, Params.char_emb_size]),trainable=True, name="char_embeddings")
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
            self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
            self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

        # Embed the question and passage information for word and character tokens
        self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
                                        self.passage_c,
                                        word_embeddings = self.word_embeddings,
                                        char_embeddings = self.char_embeddings,
                                        scope = "passage_embeddings")
        self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
                                        self.question_c,
                                        word_embeddings = self.word_embeddings,
                                        char_embeddings = self.char_embeddings,
                                        scope = "question_embeddings")

        self.passage_char_encoded = bidirectional_GRU(self.passage_char_encoded,
                                self.passage_c_len,
                                cell_fn = SRUCell if Params.SRU else GRUCell,
                                scope = "passage_char_encoding",
                                output = 1,
                                is_training = self.is_training)
        self.question_char_encoded = bidirectional_GRU(self.question_char_encoded,
                                self.question_c_len,
                                cell_fn = SRUCell if Params.SRU else GRUCell,
                                scope = "question_char_encoding",
                                output = 1,
                                is_training = self.is_training)
        self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded),axis = 2)
        self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded),axis = 2)

        # Passage and question encoding
        #cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.passage_encoding = bidirectional_GRU(self.passage_encoding,
                                self.passage_w_len,
                                cell_fn = SRUCell if Params.SRU else GRUCell,
                                layers = Params.num_layers,
                                scope = "passage_encoding",
                                output = 0,
                                is_training = self.is_training)
        #cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.question_encoding = bidirectional_GRU(self.question_encoding,
                                self.question_w_len,
                                cell_fn = SRUCell if Params.SRU else GRUCell,
                                layers = Params.num_layers,
                                scope = "question_encoding",
                                output = 0,
                                is_training = self.is_training)

    def attention_match_rnn(self):
        # Apply gated attention recurrent network for both query-passage matching and self matching networks
        with tf.variable_scope("attention_match_rnn"):
            memory = self.question_encoding
            inputs = self.passage_encoding
            scopes = ["question_passage_matching", "self_matching"]
            params = [(([self.params["W_u_Q"],
                    self.params["W_u_P"],
                    self.params["W_v_P"]],self.params["v"]),
                    self.params["W_g"]),
                (([self.params["W_v_P_2"],
                    self.params["W_v_Phat"]],self.params["v"]),
                    self.params["W_g"])]
            for i in range(2):
                args = {"num_units": Params.attn_size,
                        "memory": memory,
                        "params": params[i],
                        "self_matching": False if i == 0 else True,
                        "memory_len": self.question_w_len if i == 0 else self.passage_w_len,
                        "is_training": self.is_training,
                        "use_SRU": Params.SRU}
                cell = [apply_dropout(gated_attention_Wrapper(**args), size = inputs.shape[-1], is_training = self.is_training) for _ in range(2)]
                inputs = attention_rnn(inputs,
                            self.passage_w_len,
                            Params.attn_size,
                            cell,
                            scope = scopes[i])  # first cycle, it is "vtP"
                
                if i == 0: self.v_P = inputs
                
                memory = inputs # self matching (attention over itself)

                if i == 1: self.attn_passage = inputs
            self.self_matching_output = inputs

    def bidirectional_readout(self):
        self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output,
                                    self.passage_w_len,
                                    cell_fn = SRUCell if Params.SRU else GRUCell,
                                    # layers = Params.num_layers, # or 1? not specified in the original paper
                                    scope = "bidirectional_readout",
                                    output = 0,
                                    is_training = self.is_training)

    def pointer_network(self):
        params = (([self.params["W_u_Q"],self.params["W_v_Q"]],self.params["v"]),
                ([self.params["W_h_P"],self.params["W_h_a"]],self.params["v"]))
        cell = apply_dropout(GRUCell(Params.attn_size*2), size = self.final_bidirectional_outputs.shape[-1], is_training = self.is_training)
        self.points_logits, self.r_Q = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding, self.question_w_len, cell, params, scope = "pointer_network")


    def passage_rank(self):
        params = ([self.params["W_v_P_3"],
                self.params["W_v_Q_2"]], self.params["v_2"])   # maybe v2
        print("v_P=======>", self.v_P)
        print("r_Q=======>", self.r_Q)
        self.r_P = passage_pooling(self.v_P, self.r_Q, units = Params.attn_size, weights = params, memory_len=self.passage_w_len)
        print("=========r_P=========>", self.r_P)
        g = tf.tanh(tf.matmul(tf.concat([self.r_Q, self.r_P], 1), self.params["W_g_2"])) * self.params["v_g"]
        print("======g============>", g)

        hidden = tf.matmul(g, self.params["W_f_h"]) + self.params["b_f_h"]
        output = tf.matmul(hidden, self.params["W_f_o"]) + self.params["b_f_o"]
        #self.g_hat = tf.nn.softmax(output)
        self.g_hat = tf.tanh(output)
        print("===========g_hat====>", self.g_hat)

    def p_rank_loss(self):
        with tf.variable_scope("loss_rank"):
            shapes = self.passage_w.shape
            print(self.tags)
            self.mean_loss_p = cross_entropy_p(self.g_hat, tf.cast(self.tags, dtype=tf.float32))
            print("mean_loss_p", self.mean_loss_p)
            self.optimizer_p = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])
            
            if Params.clip:
                # gradient clipping by norm
                gradients, variables = zip(*self.optimizer_p.compute_gradients(self.mean_loss_p))
                gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
                self.train_op_p = self.optimizer_p.apply_gradients(zip(gradients, variables), global_step = self.global_step)
            else:
                self.train_op_p = self.optimizer_p.minimize(self.mean_loss_p, global_step = self.global_step)

        """
        args = {"num_units": Params.attn_size,
                "memeory": 1
        }

        # r_Q
        initial_state = question_pooling(question, units = Params.attn_size, weights = weights_q, memory_len = question_len, scope = "question_pooling")
        r_P = attention question_pooling(self.attn_passage)
        # 注意不同的长度，稍后需要mask
        g = v_g * tf.tanh(W_g * tf.concat(r_Q, r_P))
        g_hat = tf.softmax(g)
        
        cross_entropy
        total_loss = cross_entropy + loss_AP
    """

    def outputs(self):
        self.logit_1, self.logit_2 = tf.split(self.points_logits, 2, axis = 1)
        self.logit_1 = tf.transpose(self.logit_1, [0, 2, 1])
        self.dp = tf.matmul(self.logit_1, self.logit_2)
        self.dp = tf.matrix_band_part(self.dp, 0, 15)
        self.output_index_1 = tf.argmax(tf.reduce_max(self.dp, axis = 2), -1)
        self.output_index_2 = tf.argmax(tf.reduce_max(self.dp, axis = 1), -1)
        self.output_index = tf.stack([self.output_index_1, self.output_index_2], axis = 1)
        # self.output_index = tf.argmax(self.points_logits, axis = 2)

    def loss_function(self):
        with tf.variable_scope("loss"):
            shapes = self.passage_w.shape
            self.indices_prob = tf.one_hot(self.indices, shapes[1])
            print("====indices_prob===>", self.indices_prob)
            self.mean_loss = cross_entropy(self.points_logits, self.indices_prob)
            self.optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])

            if Params.clip:
                # gradient clipping by norm
                gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step = self.global_step)
            else:
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

    def summary(self):
        self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="F1")
        self.F1_placeholder = tf.placeholder(tf.float32, shape = (), name = "F1_placeholder")
        self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="EM")
        self.EM_placeholder = tf.placeholder(tf.float32, shape = (), name = "EM_placeholder")
        self.dev_loss = tf.Variable(tf.constant(5.0, shape=(), dtype = tf.float32),trainable=False, name="dev_loss")
        self.dev_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "dev_loss")
        self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.dev_loss, self.dev_loss_placeholder))
        tf.summary.scalar('loss_training', self.mean_loss)
        tf.summary.scalar('loss_dev', self.dev_loss)
        tf.summary.scalar("F1_Score",self.F1)
        tf.summary.scalar("Exact_Match",self.EM)
        tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
        #tf.summary.scalar('rank_loss', self.mean_loss_p)
        # tf.summary.histogram('g_hat', self.g_hat)
        self.merged = tf.summary.merge_all()

def debug():
    model = Model(is_training = False)
    print("Built model")

def test():
    model = Model(is_training = False); print("Built model")
    dict_ = Vocabulary()
    #dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    res = []
    with model.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
            EM, F1, Bleu_4, Rouge_L = 0.0, 0.0, 0.0, 0.0
            for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                index, ground_truth, passage, ids = sess.run([model.output_index, model.indices, model.passage_w, model.ids])
                for batch in range(Params.batch_size):
                    f1, em, bleu, rouge = f1_and_EM_bleu_rouge(index[batch], ground_truth[batch], passage[batch], dict_)
                    F1 += f1
                    EM += em
                    Bleu_4 += bleu
                    Rouge_L += rouge
                    res.append({"question_id": int(ids[batch][0]), "passage_id": int(ids[batch][1]), "spans": list(index[batch]), "passage": dict_.ind2word(passage[batch][0:20])})
            F1 /= float(model.num_batch * Params.batch_size)
            EM /= float(model.num_batch * Params.batch_size)
            Bleu_4 /= float(model.num_batch * Params.batch_size)
            Rouge_L /= float(model.num_batch * Params.batch_size)
            print("\nExact_match: {}\nF1_score: {}\nBleu_4_score:{}\nRouge_L_score:{}".format(EM,F1,Bleu_4,Rouge_L))
            
    with open(Params.outputdir, "w") as fp:
        fp.write("\n".join([json.dumps(r, ensure_ascii=False) for r in res]))

def gen_prank():
    model = Model(is_training = False); print("Built model")
    dict_ = Vocabulary()
    res = []
    with model.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir_rank))
            for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                score, passage, question, tags, ids = sess.run([model.g_hat, model.passage_w, model.question_w, model.tags, model.ids])
                for batch in range(Params.batch_size):
                    res.append({
                        "question_id": int(ids[batch][0]),
                        "passage": dict_.ind2word(passage[batch]),
                        "question": dict_.ind2word(question[batch]),
                        "score": int(score[batch][0]),
                    })
    with open("scores_rank.stat", "w") as f:
        f.write("\n".join([json.dumps(r, ensure_ascii=False) for r in res]))


def main():
    model = Model(is_training = True); print("Built model")
    #dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
    dict_ = Vocabulary()
    init = False
    devdata, dev_ind = get_dev()
    if not os.path.isfile(os.path.join(Params.logdir,"checkpoint")):
        init = True
        #glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
        #glove = np.reshape(glove,(Params.vocab_size,Params.emb_size))
        glove = np.load("../word_emb.npy")
       
    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=Params.logdir,
                        save_model_secs=0,
                        global_step = model.global_step,
                        init_op = model.init_op)
        with sv.managed_session(config = config) as sess:
            if init: sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove})
            for epoch in range(1, Params.num_epochs+1):
                if sv.should_stop(): break
                for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(model.train_op)
                    if step % Params.save_steps == 0:
                        gs = sess.run(model.global_step)
                        sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
                        sample = np.random.choice(dev_ind, Params.batch_size)
                        feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
                        index, dev_loss = sess.run([model.output_index, model.mean_loss], feed_dict = feed_dict)
                        F1, EM = 0.0, 0.0
                        Bleu_4, Rouge_L = 0.0, 0.0
                        for batch in range(Params.batch_size):
                            f1, em, bleu, rouge = f1_and_EM_bleu_rouge(index[batch], devdata[8][sample][batch], devdata[0][sample][batch], dict_)
                            F1 += f1
                            EM += em
                            Bleu_4 += bleu
                            Rouge_L += rouge
                        F1 /= float(Params.batch_size)
                        EM /= float(Params.batch_size)
                        Bleu_4 /= float(Params.batch_size)
                        Rouge_L /= float(Params.batch_size)
                        sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM, model.dev_loss_placeholder: dev_loss})
                        print("\nDev_loss: {}\nDev_Exact_match: {}\nDev_F1_score: {}\nBleu_4_score:{}\nRouge_L_score:{}".format(dev_loss,EM,F1,Bleu_4,Rouge_L))

def gen_ans():
    pass

def rank():
    model = Model(is_training = True); print("Built model")
    devdata, dev_ind = get_dev()    
    with model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=Params.logdir_rank,
                                save_model_secs=0,
                                global_step = model.global_step,
                                init_op = model.init_op)
        glove = np.load("../word_emb.npy")        
        with sv.managed_session(config=config) as sess:
            for epoch in range(1, Params.num_epochs + 1):
                if sv.should_stop(): break
                for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
                    _, g_hat, tags = sess.run(model.train_op_p, model.g_hat, model.tags)
                    print("g_hat: ", g_hat)
                    print("tags: ", tags)
                    if step % Params.save_steps == 0:
                        gs = sess.run(model.global_step)
                        sv.saver.save(sess, Params.logdir_rank + '/test_prank_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
                        sample = np.random.choice(dev_ind, Params.batch_size)
                        feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
                        loss = sess.run(model.mean_loss_p, feed_dict=feed_dict)
                        print("\n\n Dev_loss: {}".format(loss))
                    

if __name__ == '__main__':
    if Params.mode.lower() == "debug":
        print("Debugging...")
        debug()
    elif Params.mode.lower() == "test":
        print("Testing on dev set...")
        test()
    elif Params.mode.lower() == "demo":
        print("Run the local host for online demo...")
        model = Model(is_training = False, demo = True); print("Built model")
        demo_run = Demo(model)
    elif Params.mode.lower() == "train":
        print("Training...")
        main()
    elif Params.mode.lower() == "gen_ans":
        print("Generate Answer...")
        #gen_ans()
    elif Params.mode.lower() == "gen_rank":
        print("Generate Rank Scores...")
        gen_prank()
        #test()
    elif Params.mode.lower() == "prank":
        print("Passage Rank Train")
        rank()
    else:
        print("Invalid mode.")
