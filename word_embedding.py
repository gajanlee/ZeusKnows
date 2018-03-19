#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import json, math
from collections import Counter
'''
This python file implementes the word embedding model.
We use the skip-gram, infer context words from one input word.
'''


import logging
logging.basicConfig(level=logging.DEBUG,
                datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--vocabulary", default="../DuReader/data/processed/trainset/zhidao.train.json", type=str)
parser.add_argument("--endline", default=None)
parser.add_argument("--vocabulary", default="./word.dict", type=str)
parser.add_argument("--epoch", default="1", type=int)
parser.add_argument("--passages", default="./word_list.dict")
args = parser.parse_args()



class Word2Vec(object):
    """Word2Vec model (Skip-gram)."""
    def __init__(self, sess):
        self.word_list = []
        self._id_to_vocab = {}
        self._vocab_to_id = {}

        #self.window_size = 2
        self.batch_size = 300
        self.num_sampled = 100
        self.vocab_size = None
        self.emb_dim = 300
        self.window_size = 4
        self.sess = sess

        self.vocab_dict = {}
        
        self.rd = self.reader()

    def reader(self):
        with open(args.passages) as fp:
            for line in fp:
                try:
                    yield from self.target_window([int(x) for x in line.split(' ')])
                except:
                    pass

    # id 0 is <unknown>
    def target_window(self, id_list):
        targets = []    # a list of (inputs, label)
        for idx, label in enumerate(id_list):
            if label == 0: continue
            target_window_size = np.random.randint(1, self.window_size + 1)
            # 这里要考虑input word前面单词不够的情况
            start_point = idx - target_window_size if (idx - target_window_size) > 0 else 0
            end_point = idx + target_window_size
            for input in set(id_list[start_point: idx] + id_list[idx+1: end_point+1]):
                if input != 0: targets.append((input, label))
        return targets

    def get_batch(self):
        inputs, labels = [], []
        while len(inputs) != self.batch_size:
            try:
                input, label = next(self.rd)
                inputs.append(input), labels.append([label])
            except StopIteration:
                return None, None
        return inputs, labels
        

    # Generate a token dictionary.
    """def preprocess_word_list(self, data_dict):
        for doc in data_dict["documents"]:
            for para in doc["segmented_paragraphs"]:
                for word in para:
                    self.word_list.append(word)
    
    def vocab_reader(self, filepaths, start_line=0, end_line=None):
        for filepath in filepaths:
            with open(filepath) as f:
                for _ in range(start_line): next(f)
                for i, l in enumerate(f):
                    if end_line is not None and i+start_line == end_line: break
                    self.preprocess_word_list(json.loads(l))
        self.arrange()
    
    def arrange(self):
        vocab = set(self.word_list)
        vocab.add("<unknown>")
        self._vocab_to_id = {w: c for c, w in enumerate(vocab)}
        self._id_to_vocab = {c: w for c, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def vocab_to_id(self, word):
        word = word if word in self._vocab_to_id else "<unknown>"
        return self._vocab_to_id[word]
    
    def vocab_saver(self, filepath="vocab.dict"):
        with open(filepath, "w") as f:
            for word, id in self._vocab_to_id.items():
                f.write("{word} {id}\n".format(word=word, id=id))

    def get_context(self, idx):
        target_window = np.random.randint(1, self.window_size + 1)
        # 这里要考虑input word前面单词不够的情况
        start_point = idx - target_window if (idx - target_window) > 0 else 0
        end_point = idx + target_window
        targets = set(self.word_list[start_point: idx] + self.word_list[idx+1: end_point+1])  # xiaochu chongfu
        return list(targets)

    def generate_batch(self):
        for idx in range(len(self.word_list)):
            for word in self.get_context(idx):
                yield self.vocab_to_id(self.word_list[idx]), self.vocab_to_id(word)
    
    def next_batch(self):
        g_batch = self.generate_batch()

        def batch():
            inputs, labels = [], []
            for _ in range(self.batch_size):
                try:
                    ip, lb = next(g_batch)
                    # raise StopIteration. and return None
                    inputs.append(ip); labels.append([lb])
                except StopIteration:
                    return None, None
            return inputs, labels
        return batch

    def get_batch(self):
        with open() as fp:
            pass"""



    def build_graph(self):
        self.vocab_size = len(self.vocab_dict)

        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        embeddings = tf.Variable(
            tf.random_uniform([self.vocab_size, self.emb_dim], -1.0, 1.0), name="embeddings")
        self._emb = embeddings

        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocab_size, self.emb_dim], stddev=1.0 / math.sqrt(self.emb_dim)), name="weights")
        nce_biases = tf.Variable(tf.zeros([self.vocab_size]), name="nce_bias")

        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
        
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, self.train_labels, embed, self.num_sampled, self.vocab_size))
        
        self.loss = loss
        tf.summary.scalar("loss", loss)
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        self.merge_sumarry_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("/tmp/emb_logs", self.sess.graph)
    
    def build_eval_graph(self):
        """Build the Eval Graph."""
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)
        pass
    
    def restore(self):
        self.saver.restore(self.sess, "./model.ckpt-1")

    def train(self, epoch):
        count = 0
        while True:
            inputs, labels = self.get_batch()
            if inputs is None: break
            _, cur_loss, merge = self.sess.run([self.trainer, self.loss, self.merge_sumarry_op], feed_dict={self.train_inputs: inputs, self.train_labels: labels})
            count += 1
            if count % 1000 == 0:
                self.summary_writer.add_summary(merge, count)
                print("batch: %s, loss: %s" % (count, cur_loss))
        self.saver.save(self.sess, "./model_check/embedding.ckpt", global_step=epoch)

def load_vocab_dict():
    vocab_dict = {}
    with open(args.vocabulary) as fp:
        for line in fp:
            res = line.split(' ')
            vocab_dict[res[1]] = res[0]   # content: id
    return vocab_dict

def main():
    #self.vocab_reader(["data/preprocessed/trainset/zhidao.train.json"], end_line=10000)
    #self.vocab_saver()
    #print("save done")
    with tf.Graph().as_default(), tf.Session() as sess:
        #self.saver.restore(self.sess, "./model.ckpt-1")
        
        model = Word2Vec(sess)
        logger.info("loading vocab")
        model.vocab_dict = load_vocab_dict()
        logger.info("loading vocab done, vocab_size is %s" % len(model.vocab_dict))
        model.build_graph()
        logger.info("build graph done")
        for i in range(args.epoch):
            logger.info("%s epoch starts..." % i)
            model.train(i)

def main_test():
    model = Word2Vec(None)
    count = 0
    while True:
        inputs, labels = model.get_batch()
        count += 1
        if count % 1000 == 0:
            print(count)
        if inputs is None:
            break
if __name__ == "__main__":
    main_test()

