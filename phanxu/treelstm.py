import codecs
import numpy as np
import tensorflow as tf
import tensorflow_fold.public.blocks as td

from common import conllu


class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh,
                 keep_prob=1.0, seed=None):
        """Initialize the cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      keep_prob: Keep probability for recurrent dropout.
      seed: Random seed for dropout.
    """
        super(BinaryTreeLSTMCell, self).__init__(
            num_units, forget_bias=forget_bias, activation=activation)
        self._keep_prob = keep_prob
        self._seed = seed

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # ci, hi
            num_children = len(state)
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, sum([ch[1] for ch in state])], 1), 3 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, o = tf.split(value=concat, num_or_size_splits=3, axis=1)

            concat_f = tf.contrib.layers.linear(
                tf.concat([inputs] + [ch[1] for ch in state], 1), num_children * self._num_units)

            f = tf.split(value=concat_f, num_or_size_splits=num_children, axis=1)

            j = self._activation(j)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j = tf.nn.dropout(j, self._keep_prob, seed=self._seed)

            new_c = tf.sigmoid(i) * j
            for idx, (ci, hi) in state:
                new_c += ci * tf.sigmoid(f[idx] + self._forget_bias)

            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def load_trees(raw1, raw2, gold=None):
    def traverse(_sen, head):
        nodes = []
        for __id in _sen:
            if _sen[__id][conllu.HEAD] == head:
                nodes.append(traverse(_sen, __id))
        return _sen[head][conllu.FORM], tuple(nodes)

    def preprocess(sen):
        # fix multiple roots issue in bst by choosing 1st word to be root
        root = None
        for i in sen:
            if sen[i][conllu.HEAD] == '0':
                if root is None:
                    root = i
                else:
                    sen[i][conllu.HEAD] = root
        return root

    def score(sen_data, sen_gold):
        val_ids = [_id for _id in sen_gold if _id != '0' and '-' not in _id and '.' not in _id]
        return sum([sen_data[_id][conllu.HEAD] == sen_gold[_id][conllu.HEAD] for _id in val_ids]) * 1.0 / len(val_ids)

    if gold is None:
        gold = [None] * len(raw1)

    res = []
    vocab = set([])

    for (_, sen1), (_, sen2), (_, s_gold) in zip(raw1, raw2, gold):
        root1 = preprocess(sen1)
        root2 = preprocess(sen2)
        _score = score(sen1, s_gold) > score(sen2, s_gold) if s_gold is not None else None
        for w in s_gold:
            vocab.add(s_gold[w][conllu.FORM])
        res.append(((traverse(sen1, root1), traverse(sen2, root2)), _score))

    return res, vocab


def load_embeddings(filename, vocab):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    weight_vectors = []
    word_idx = {}
    with codecs.open(filename, encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            if line_no < 1:
                continue
            try:
                vec = line.strip().split(' ')
                if len(weight_vectors) > 0:
                    assert len(vec[1:]) == weight_vectors[0].shape[0], '%s %s' % (len(vec[1:]), weight_vectors[0].shape[0])
                if vec[0] not in vocab:
                    continue
                weight_vectors.append(np.array(vec[1:], dtype=np.float32))
                word_idx[vec[0]] = len(weight_vectors)
            except Exception as e:
                print('Embedding error on line %d: %s' % (line_no, e))
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(
        -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    return np.stack(weight_vectors), word_idx


def create_embedding(weight_matrix):
    return td.Embedding(*weight_matrix.shape, initializer=weight_matrix,
                        name='word_embedding')


def create_model(word_embedding, word_idx, lstm_num_units, mlp_size, keep_prob=1):
    """Creates a sentiment model. Returns (compiler, mean metrics)."""
    tree_lstm = td.ScopedLayer(
        tf.contrib.rnn.DropoutWrapper(
            BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob),
            input_keep_prob=keep_prob, output_keep_prob=keep_prob),
        name_or_scope='tree_lstm')

    embed_subtree = td.ForwardDeclaration(output_type=tree_lstm.state_size)

    unknown_idx = len(word_idx)

    def lookup_word(word):
        return word_idx.get(word, unknown_idx)

    word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                td.Scalar('int32') >> word_embedding)
    child2vec = td.GetItem(1) >> td.TupleType(td.Map(embed_subtree()))

    tree2vec = td.AllOf(word2vec, child2vec)

    tree = tree2vec >> tree_lstm

    embed_subtree.resolve_to(tree)

    expression_logits = (td.GetItem(0) >> td.Map(tree) >> td.Concat()
                         >> td.FC(mlp_size, activation='relu', name='mlp1')
                         >> td.FC(mlp_size, activation='relu', name='mlp2'))

    expression_label = td.GetItem(1) >> td.Scalar('int32')

    model = td.AllOf(expression_logits, expression_label)

    compiler = td.Compiler.create(model)

    logits, labels = compiler.output_tensors

    _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))

    _accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1),
                         tf.argmax(logits, 1)),
                dtype=tf.float32))

    return compiler, _loss, _accuracy
