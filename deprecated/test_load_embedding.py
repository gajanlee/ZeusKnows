"""
---Deprecated

StackOverflow: How to load tensorflow model correctly.
Load tensorflow saved model example.
"""

import tensorflow as tf
sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model_check/embedding.ckpt-15.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model_check'))
emb = sess.run('embeddings:0')
print(emb[10][0])
import numpy as np
np.save("word_emb.npy", emb)
print("Save Over")
b = np.load("word_emb.npy")
print(b.shape)
print(b[10][0])
"""# Access saved Variables directly
#print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
#graph = tf.get_default_graph()
#w1 = graph.get_tensor_by_name("w1:0")
#w2 = graph.get_tensor_by_name("w2:0")
#feed_dict ={w1:13.0,w2:17.0}
#Now, access the op that you want to run. 
#op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#print sess.run(op_to_restore,feed_dict)
"""
