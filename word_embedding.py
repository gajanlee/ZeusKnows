from __init__ import *
import tensorflow as tf
import numpy as np
import math
from params import Params


class Word2Vec(object):
    """Word2Vec model (Skip-gram)."""
    def __init__(self, sess):
        self.batch_size = 3000
        self.num_sampled = 100
        self.vocab_size = None
        self.emb_dim = 300
        self.window_size = 4
        self.sess = sess
        
        self.rd = self.reader()

    def reader(self):
        with open("_id" + Params.wordlst_path) as fp:
            for line in fp:
                try: yield from self.target_window([int(x) for x in line.split(' ')])
                except: print(line)

    def target_window(self, id_list):
        targets = []
        for idx, label in enumerate(id_list):
            if label == 0: continue # delete unknown label and input(later)
            target_window_size = np.random.randint(1, self.window_size + 1)
            # 这里要考虑input word前面单词不够的情况
            start_point = idx - target_window_size if (idx - target_window_size) > 0 else 0
            for input in set(id_list[start_point: idx] + id_list[idx+1: idx+target_window_size +1]):
                if input != 0: targets.append((input, label))
        return targets

    def get_batch(self):
        inputs, labels = [], []
        while len(inputs) != self.batch_size:
            try:
                input, label = next(self.rd)
                inputs.append(input), labels.append([label])
            except StopIteration: return None, None
        return inputs, labels

    def build_graph(self):
        # dict contains some space like char.
        self.vocab_size = len(vocabulary.vocab_dict)
        self.vocab_size = 144836
        logger.info("BUILD GRAPH: VOCABULARY SIZE IS %s" % len(vocabulary.vocab_dict))

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
    
    def restore(self):
        self.saver.restore(self.sess, "./model.ckpt-1")

    def train(self, epoch):
        count = 0; self.rd = self.reader()
        while True:
            inputs, labels = self.get_batch()
            if inputs is None: break
            _, cur_loss, merge = self.sess.run([self.trainer, self.loss, self.merge_sumarry_op], feed_dict={self.train_inputs: inputs, self.train_labels: labels})
            count += 1
            if count % 1000 == 0:
                self.summary_writer.add_summary(merge, count)
                logger.info("batch: %s, loss: %s" % (count, cur_loss))
        self.saver.save(self.sess, "./model_check/embedding.ckpt", global_step=epoch)

    def build_eval_graph(self):
        """Build the Eval Graph."""
        saver = tf.train.import_meta_graph('./model_check/embedding.ckpt-15.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./model_check'))
        emb = self.sess.run('embeddings:0')
        
        nemb = tf.nn.l2_normalize("embeddings:0", 1)
        
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)

        a_emb = tf.gather(nemb, analogy_a)
        b_emb = tf.gather(nemb, analogy_b)
        c_emb = tf.gather(nemb, analogy_c)

        target = c_emb + (a_emb - b_emb)
        dist = tf.matmul(target, nemb, transpose_b=True)
        _, pred_idx = tf.nn.top_k(dist, 4)

        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemb, nearby_word)
        nearyby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearyby_dist, min(1000, nemb.shape[0]))


        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def eval(self):
        self.nearby(["苹果", "一", "手机"])
    
    def nearby(self, words, num=20):
        """Prints out nearby words given a list of words."""
        ids = np.array([vocabulary.getVocabID(x) for x in words])
        vals, idx = self.sess.run(
            [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in range(len(words)):
            print("\n%s\n=====================================" % (words[i]))
        for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
            print("%-20s %6.4f" % (vocabulary.VocabID_to_vocab(neighbor), distance))


def main():
    with tf.Graph().as_default(), tf.Session() as sess:
        model = Word2Vec(sess)
        model.build_graph()
        logger.info("build graph done")
        # model.saver.restore(sess, "./model_check/embedding.ckpt-0")
        logger.info("load chekpoint done")
        for i in range(Params.epoch):
            logger.info("%s epoch starts..." % i)
            model.train(i + 1)

def eval_main():
    with tf.Graph().as_default(), tf.Session as sess:
        model = Word2Vec(sess)
        model.build_eval_graph()
        logger.info("build evaluate graph done")

        model.eval()

if __name__ == "__main__":
    main()
