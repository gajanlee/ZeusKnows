

v = tf.variable(tf.float32, [att_size, att_size])

def attention(inputs, memory, hidden_size):
    for t in range(passage_len):
        
        s = v * tf.tanh(W_u_Q * u_j_Q + W_u_P * W_u_P  + .. + W_v_P * vtP)

        at = tf.sigmoid(st)
        ct = a1t * uiQ

        gt = tf.sigmoid(W_g * [utP, c_t])
        return gt * [utP, c_t]