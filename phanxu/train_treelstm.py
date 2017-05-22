import os.path

import pickle
import tensorflow as tf
import tensorflow_fold.public.blocks as td

import treelstm

from common import conllu

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_base', None, 'Path prefix for saving checkpoints')
flags.DEFINE_integer(
    'epochs', 20, 'Number of training epochs. Zero to run forever.')
flags.DEFINE_integer(
    'batch_size', 100, 'Number of examples per batch.')
flags.DEFINE_float(
    'learning_rate', 0.05, 'Learning rate for Adagrad optimizer.')
flags.DEFINE_float(
    'embedding_learning_rate_factor', 0.1,
    'Scaling factor for gradient updates to word embedding vectors.')
flags.DEFINE_float(
    'keep_prob', 0.75, 'Keep probability for dropout.')
flags.DEFINE_string(
    'tree_1_dir', None, 'Directory for trees (train.conllu, dev.conllu, test.conllu).')
flags.DEFINE_string(
    'tree_2_dir', None, 'Directory for trees (train.conllu, dev.conllu, test.conllu).')
flags.DEFINE_string(
    'tree_gold_dir', None, 'Directory for trees (train.conllu, dev.conllu, test.conllu).')
flags.DEFINE_string(
    'embedding_file', None, 'File name for word embeddings.')
flags.DEFINE_integer(
    'lstm_num_units', 300, 'Number of units for tree LSTM.')
flags.DEFINE_integer(
    'mlp_size', 300, 'Number of hidden units for MLP.')


def main(_):

    train_files = (os.path.join(FLAGS.tree_1_dir, 'train.conllu'),
                   os.path.join(FLAGS.tree_2_dir, 'train.conllu'),
                   os.path.join(FLAGS.tree_gold_dir, 'train.conllu'))
    print('loading training trees from:\n%s' % str(train_files))
    train_trees, vocab = treelstm.load_trees(conllu.CoNLLU(train_files[0]).get_content(),
                                             conllu.CoNLLU(train_files[1]).get_content(),
                                             conllu.CoNLLU(train_files[2]).get_content())

    embedding_preload = os.path.join(FLAGS.checkpoint_base, 'word_embedding.pkl')
    if os.path.exists(embedding_preload):
        print('Loading word embedding matrix from %s' % embedding_preload)
        with open(embedding_preload, 'rb') as f:
            weight_matrix, word_idx = pickle.load(f)
    else:
        print('loading word embeddings from %s' % FLAGS.embedding_file)
        weight_matrix, word_idx = treelstm.load_embeddings(FLAGS.embedding_file, vocab)
        with open(embedding_preload, 'wb') as f:
            pickle.dump((weight_matrix, word_idx), f)

    dev_files = (os.path.join(FLAGS.tree_1_dir, 'dev.conllu'),
                 os.path.join(FLAGS.tree_2_dir, 'dev.conllu'),
                 os.path.join(FLAGS.tree_gold_dir, 'dev.conllu'))
    print('loading dev trees from:\n%s' % str(dev_files))
    dev_trees, _ = treelstm.load_trees(conllu.CoNLLU(dev_files[0]).get_content(),
                                       conllu.CoNLLU(dev_files[1]).get_content(),
                                       conllu.CoNLLU(dev_files[2]).get_content())

    with tf.Session() as sess:
        print('creating the model')
        keep_prob = tf.placeholder_with_default(1.0, [])
        train_feed_dict = {keep_prob: FLAGS.keep_prob}
        word_embedding = treelstm.create_embedding(weight_matrix)
        compiler, loss, accuracy = treelstm.create_model(
            word_embedding, word_idx, FLAGS.lstm_num_units, FLAGS.mlp_size, keep_prob)
        opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)
        grads_and_vars = opt.compute_gradients(loss)
        found = 0
        for i, (grad, var) in enumerate(grads_and_vars):
            if var == word_embedding.weights:
                found += 1
                grad = tf.scalar_mul(FLAGS.embedding_learning_rate_factor, grad)
                grads_and_vars[i] = (grad, var)
        assert found == 1  # internal consistency check
        train = opt.apply_gradients(grads_and_vars)
        saver = tf.train.Saver()

        print('initializing tensorflow')
        sess.run(tf.global_variables_initializer())

        with compiler.multiprocessing_pool():
            print('training the model')
            train_set = compiler.build_loom_inputs(train_trees)
            dev_feed_dict = compiler.build_feed_dict(dev_trees)
            dev_accuracy_best = 0.0
            for epoch, shuffled in enumerate(td.epochs(train_set, FLAGS.epochs), 1):
                train_loss = 0.0
                for batch in td.group_by_batches(shuffled, FLAGS.batch_size):
                    train_feed_dict[compiler.loom_input_tensor] = batch
                    _, batch_loss, batch_accuracy = sess.run([train, loss, accuracy], train_feed_dict)
                    train_loss += batch_loss
                dev_loss, dev_accuracy = sess.run([loss, accuracy], dev_feed_dict)
                print('epoch:%4d, train_loss: %.3e, dev_loss: %.3e, dev_accuracy: [%s]'
                      % (epoch, train_loss, dev_loss, ' '.join(dev_accuracy)))

                if dev_accuracy > dev_accuracy_best:
                    dev_accuracy_best = dev_accuracy
                    save_path = saver.save(sess, FLAGS.checkpoint_base, global_step=epoch)
                    print('model saved in file: %s' % save_path)


if __name__ == '__main__':
    tf.app.run()
