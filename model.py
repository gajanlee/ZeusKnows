import tensorflow as tf

class Model(object):
    def __init__(self):
        self.word_embedding = tf.Variable(initializer=tf.constant(
            word_embedding, dtype=tf.float32), trainable=False) # word_embedding's size is [vocab_size, vocab_dim]
        self.char_embedding = tf.Variable(tf.Zeros([char_size, char_dim]))
        self.batch_size = 300

        self.passage_words = tf.placeholder()
        self.question_words = tf.placeholder(tf.int32, shape=[batch_size, None, vocab_size]) # Actually, it should be "max_passage_length" instead of None
        self.passage_chars = tf.placeholder(tf.int32, shape=[batch_size, None, char_dim])    # 
        self.question_chars = tf.placeholder(tf.int32, shape=[batch_size, None, char_dim]) # Actually, it should be "max_passage_length" instead of None

        with tf.name_scope("encoding"):
            passage_word_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_words)
            passage_char_emb = tf.nn.embedding_lookup(self.char_embedding, self.passage_chars)
            question_word_emb = tf.nn.embedding_lookup(self.word_embedding, self.question_words)
            question_char_emb = tf.nn.embedding_lookup(self.char_embedding, self.question_chars)


            passage_emb = tf.concat((passage_word_emb, passage_char_emb)), axis=2)
            question_emb = tf.concat((question_word_emb, question_char_emb)), axis=2)
            

        with tf.variable_scope("encoding"):
            # First Bidirectional RNN is used to generate char embeddings for passages and questions.
            self.passage_char_encoded = bidirectional_GRU()
            self.question_char_encoded = bidirectional_GRU()
            
            # Now we get the paper's [etQ, ctQ]
            passage_encoding = tf.concat((passage_word_emb, passage_word_emb), axis=2)
            question_encoding = tf.concat((question_word_emb, question_char_emb), axis=2)

            # Now GRU to get utQ and utP
            passage_encoding = bidirectional_GRU(passage_encoding, )
            question_encoding = bidirectional_GRU(question_encoding)

        attention_size = 75
        initializer = tf.truncated_normal_initializer
        W_u_Q = ;
        W_u_P = ;
        W_v_P = ;
        W_g = tf.get_variable("W_g", dtype = tf.float32, shape=(4*attn_size, 4*attn_size), initializer=initializer())
        
        with tf.variable_scope("attention"):
            inputs = passage_encoding
            memory = question_encoding
            # drop out inputs and memory
            for scope in ["gated_attention", "self_attention"]:
                with tf.variable_scope(scope):
                    # gated_attention's hidden layer is attention_size
                    # and memory is questions' encoding
                    cell_fw = ;


                    cell_fw， cell_bw = [tf.nn.rnn_cell.GRUCell(attention_size, activation=tanh) for _ in range(2)]
                    

                    RNN()

                    inputs = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length = passage_words_length, dytpe=tf.float32)
                    memory = inputs

                    


                    # inputs is utP, we need [utP, ct]
                    inputs = tf.concat((inputs, ct), axis = 1)
                    g_t = tf.sigmoid(tf.matmul(inputs, W_g))
                    new_inputs = g_t * inputs
                    
            
            self.self_matching_output = inputs

            




class gate_attention_RNN_Wrapper(RNNCell):
    def __init__(self, num_units, memory):
        super(gated_attention_Wrapper, self).__init__(_reuse=reuse)
        self._cativation = math_ops.tanh
        self._num_units = num_units
        self._attention = memory
        cell = 

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
    
    def __call__(self, inputs, state, scope=None):
        inputs = gated_attention()

def gated_attention(memory, units, weights, memory_len):
    with tf.variable_scope("question_pooling"):
        shapes = memory.get_shape().as_list()
        

















class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
			   is_training = True):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._is_training = is_training

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope = None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if inputs.shape.as_list()[-1] != self._num_units:
        with vs.variable_scope("projection"):
            res = linear(inputs, self._num_units, False, )
    else:
        res = inputs
    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(
          linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(
          linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer))
    #   recurrent dropout as proposed in https://arxiv.org/pdf/1603.05118.pdf (currently disabled)
      #if self._is_training and Params.dropout is not None:
        #c = tf.nn.dropout(c, 1 - Params.dropout)
    new_h = u * state + (1 - u) * c
    return new_h + res, new_h


def linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)


class gated_attention_Wrapper(RNNCell):
  def __init__(self,
               num_units,
               memory,
               params,
               self_matching = False,
               memory_len = None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,):
    super(gated_attention_Wrapper, self).__init__(_reuse=reuse)
    cell = SRUCell if use_SRU else GRUCell
    self._cell
    self._cell = cell(num_units, is_training = is_training)
    self._num_units = num_units #
    self._activation = math_ops.tanh
    self._attention = memory  #
    self._params = params #
    self._self_matching = self_matching #
    self._memory_len = memory_len
    self._is_training = is_training

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope = None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope("attention_pool"):
        inputs = gated_attention(self._attention,
                                inputs,
                                state,
                                self._num_units,
                                params = self._params,
                                self_matching = self._self_matching,
                                memory_len = self._memory_len)
    output, new_state = self._cell(inputs, state, scope)
    return output, new_state



from tensorflow.contrib.rnn import MultiRNNCell
def bidirectional_GRU(embedding, char_length, scope):
    tf.variable_scope(scope):
        shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

            cell_fn = tf.contrib.rnn.GRUCell
            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(cell_fn(units), size = inputs.shape[-1], is_training = is_training) for _ in range(2)]


def apply_dropout(inputs, size = None, is_training = True):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if Params.dropout is None and Params.zoneout is None:
        return inputs
    if Params.zoneout is not None:
        return ZoneoutWrapper(inputs, state_zoneout_prob= Params.zoneout, is_training = is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = 1 - Params.dropout,
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs















    def build_graph(self):
        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape()
        
        with tf.variable_scope("encoding"):
            rnn = gru(name_layer=3)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            pass
        
        with tf.variable_scope("match"):
            pass


def encode_ids():
    self.char_embedding
            
