#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import json
from collections import Counter
import math
'''
This python file implementes the word embedding model.
We use the skip-gram, infer context words from one input word.
'''


class Word2Vec(object):
    """Word2Vec model (Skip-gram)."""
    
    def __init__(self, sess):
        self.word_list = []
        self._id_to_vocab = {}
        self._vocab_to_id = {}

        self.window_size = 2
        self.batch_size = 300
        self.num_sampled = 100
        self.vocab_size = None
        self.emb_dim = 300

        self.sess = sess


    # Generate a token dictionary.
    def preprocess_word_list(self, data_dict):
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

    def build_graph(self):
        
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

    def train(self):
        batch, count = self.next_batch(), 0
        while True:
            inputs, labels = batch()
            if inputs is None: break
            _, cur_loss, merge = self.sess.run([self.trainer, self.loss, self.merge_sumarry_op], feed_dict={self.train_inputs: inputs, self.train_labels: labels})
            count += 1
            if count % 100 == 0:
                self.summary_writer.add_summary(merge, count)
                print(count, ":", cur_loss)

    def main(self):
        self.vocab_reader(["preprocessed/trainset/zhidao.train.json"], end_line=30)
        self.vocab_saver()
        self.build_graph()
        self.saver.restore(self.sess, "./model.ckpt-1")
        for _ in range(10):
            self.train()
        self.saver.save(self.sess, "./model.ckpt", global_step=1)

if __name__ == "__main__":
    with tf.Graph().as_default(), tf.Session() as sess:
        Word2Vec(sess).main()
























def sample():
    t = 1e-5
    threshold = 0.8
    int_word_counts = Counter(word_list)
    total_count = len(word_list)
    prob_drop = {w: 1 - np.sqrt(t / (c / total_count)) for w, c in int_word_counts.items()}
    train_words = [w for w in int_word_counts if prob_drop[w] < threshold]

def get_targets(words, idx, window_size=5):
    target_window = np.random.randint(1, window_size+1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point: idx] + words[idx+1: end_point+1])
    return list(targets)

def get_batches(words, batch_size, window_size=5):
    # add <s> and </s>
    # add <unknown>
    n_batches = len(words) 
    
    # 仅取full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx+batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x]*len(batch_y))
            y.extend(batch_y)
        yield x, y

def model():
    vocab_size = len(int_to_vocab)
    embedding_size = 200
    window_size = 8
    n_sampled = 100
    
    train_graph = tf.Graph()
    with tf.name_scope("input"):
        inputs = tf.placeholder(tf.float32, [None, window_size], name="inputs")
        labels = tf.placeholder(tf.int32, [None, None], name="labels")

        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, -1))
        embed = tf.nn.embedding_lookup(embedding, inputs)
        
        softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(vocab_size))
        
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minmize(cost)
