import argparse
from collections import Counter
from os.path import exists
import pickle

import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Input
from keras.engine import Model
from keras.layers import Embedding, TimeDistributed, Dense, Lambda, Dropout, BatchNormalization, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

from config import *
from common import conllu
from common import utils


class PhanXu:

    def __init__(self, model_prefix, train_files=None, gold_file_path=None):
        self.train_data = None
        self.train_labels = None
        self.VOCAB_SIZE = 0
        self.word_embedding_matrix = None
        self.model = None
        self.word_index = None
        self.model_prefix = model_prefix
        if train_files is not None and gold_file_path is not None:
            self.read_files(train_files, gold_file_path)

    def get_vocabs(self, data):
        freqs = Counter([word[conllu.FORM]
                         for _, sentence in data.get_content() for _, word in sentence.iteritems()])
        vocab = sorted(freqs.keys(), key=freqs.get, reverse=True)[:MAX_VOCAB_SIZE]

        self.word_index = {word: i for (i, word) in enumerate(vocab)}

    def parse_data(self, data):
        res_1 = []
        res_2 = []
        for _, sentence in data:
            ids_1 = sorted([_id for _id in sentence if _id != '0' and '-' not in _id and '.' not in _id],
                           key=utils.get_id_key)
            ids_2 = [sentence[_id][conllu.HEAD] for _id in ids_1]

            res_1.append([self.word_index.get(sentence[_id][conllu.FORM], len(self.word_index)) for _id in ids_1])
            res_2.append([self.word_index.get(sentence[_id][conllu.FORM], len(self.word_index)) for _id in ids_2])
        return [pad_sequences(res_1, maxlen=MAX_LEN), pad_sequences(res_2, maxlen=MAX_LEN)]

    @staticmethod
    def score(data, gold):
        res = []
        for (sen_data, sen_gold) in zip(data, gold):
            sen_data = sen_data[1]
            sen_gold = sen_gold[1]
            val_ids = [_id for _id in sen_gold if _id != '0' and '-' not in _id and '.' not in _id]
            res.append(sum([sen_data[_id][conllu.HEAD] == sen_gold[_id][conllu.HEAD] for _id in val_ids])*1.0/len(val_ids))
        return res

    def read_files(self, train_files, gold_file_path, embedding_file_path=None):
        if (exists(self.model_prefix + '_train_data.npy') and exists(self.model_prefix + '_word_index.pkl') and
                exists(self.model_prefix + '_train_labels.npy')):
            print('Load data from files')
            self.train_data = np.load(self.model_prefix + '_train_data.npy')
            self.train_labels = np.load(self.model_prefix + '_train_labels.npy')
            self.word_index = pickle.load(open(self.model_prefix + '_word_index.pkl', 'rb'))
        else:
            print('Parse raw data')
            gold_raw = conllu.CoNLLU(gold_file_path)
            self.get_vocabs(gold_raw)

            self.train_data = []
            self.train_labels = []
            for train_file in train_files:
                train_raw = conllu.CoNLLU(train_file)
                parsed_tmp = self.parse_data(train_raw.get_content())
                if len(self.train_data) > 0:
                    self.train_data[0] = np.concatenate((self.train_data[0], parsed_tmp[0]))
                    self.train_data[1] = np.concatenate((self.train_data[1], parsed_tmp[1]))
                else:
                    self.train_data = parsed_tmp
                self.train_labels += self.score(train_raw.get_content(), gold_raw.get_content())

            print('Parsed %d in file, %d samples, %d labels' % (len(gold_raw.get_content()),
                                                                len(self.train_data[0]), len(self.train_labels)))

            np.save(self.model_prefix + '_train_data.npy', self.train_data)
            np.save(self.model_prefix + '_train_labels.npy', self.train_labels)
            pickle.dump(self.word_index, open(self.model_prefix + '_word_index.pkl', 'wb'))

        self.VOCAB_SIZE = min(len(self.word_index), MAX_VOCAB_SIZE)

        # Read pre-trained embedding
        if exists(self.model_prefix + '_word_embedding.npy'):
            print('Load word embedding matrix from file')
            self.word_embedding_matrix = np.load(self.model_prefix + '_word_embedding.npy')
        elif embedding_file_path is not None:
            print('Init word embedding matrix with pre-trained GloVe')
            self.word_embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, EMBEDDING_SIZE))

            with open(embedding_file_path, encoding='utf-8') as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    if word in self.word_index:
                        self.word_embedding_matrix[self.word_index[word]] = np.asarray(values[1:], dtype='float32')

            np.save(self.model_prefix + '_word_embedding.npy', self.word_embedding_matrix)

    def get_model(self):
        # Defining model
        print('Define model')
        q1 = Input(shape=(MAX_LEN,), dtype='int32')
        q2 = Input(shape=(MAX_LEN,), dtype='int32')

        if self.word_embedding_matrix is None:
            embed = Embedding(self.VOCAB_SIZE + 1, EMBEDDING_SIZE, input_length=MAX_LEN, trainable=TRAIN_EMBED)
        else:
            embed = Embedding(self.VOCAB_SIZE + 1, EMBEDDING_SIZE, weights=[self.word_embedding_matrix],
                              input_length=MAX_LEN, trainable=TRAIN_EMBED)
        embed_q1 = embed(q1)
        embed_q2 = embed(q2)

        translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

        sent_q1 = translate(embed_q1)
        sent_q2 = translate(embed_q2)

        sent_embed = Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))
        sent_q1 = BatchNormalization()(sent_embed(sent_q1))
        sent_q2 = BatchNormalization()(sent_embed(sent_q2))

        joint = concatenate([sent_q1, sent_q2])
        joint = Dropout(DROPOUT_RATE)(joint)

        for i in range(3):
            joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2) if L2 else None)(joint)
            joint = Dropout(DROPOUT_RATE)(joint)
            joint = BatchNormalization()(joint)

        pred = Dense(1, activation='sigmoid')(joint)

        model = Model(inputs=[q1, q2], outputs=pred)
        model.compile(optimizer=OPTIMIZER, loss='mean_squared_error', metrics=['accuracy'])

        return model

    def train(self):
        if self.model is None:
            self.model = self.get_model()

        # Training
        print('Training')
        callbacks = [EarlyStopping(patience=PATIENCE),
                     ModelCheckpoint(self.model_prefix + '.h5', save_best_only=True, save_weights_only=True)]
        self.model.fit([self.train_data[0], self.train_data[1]], self.train_labels, validation_split=VALIDATION_SPLIT,
                       epochs=NB_EPOCHS, callbacks=callbacks)

        self.model.load_weights(self.model_prefix + '.h5')

    def test(self, input_files, output_file):
        datas = []
        preds = []
        for input_file in input_files:
            datas.append(conllu.CoNLLU(input_file).get_content())
            preds.append(self.model.predict(self.parse_data(datas[-1])).flatten())

        preds = np.array(preds).transpose()
        res = conllu.CoNLLU()
        for _id, row in enumerate(preds):
            print(row)
            pred_max = np.argmax(row)
            res._content.append(datas[pred_max][_id])

        res.to_file(output_file)

    def load_model(self):
        self.model = self.get_model()
        print('Load weights from file')
        self.model.load_weights(self.model_prefix + '.h5')

        if self.word_index is None:
            print('Load word_index from file')
            self.word_index = pickle.load(open(self.model_prefix + '_word_index.pkl', 'rb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("-m", "--model-prefix", default='./saves/phanxu/grc-ud')
    parser.add_argument("-i", "--train-data", action='append')
    parser.add_argument("-g", "--gold-data", default='./data/treebanks/grc-ud-train.conllu')
    parser.add_argument("-t", "--test-data", action='append')
    parser.add_argument("-o", "--output", default='./saves/phanxu/grc-ud-test.conllu')
    args = parser.parse_args()

    if args.task == 'all':
        phanxu = PhanXu(model_prefix=args.model_prefix, train_files=args.train_data, gold_file_path=args.gold_data)
        phanxu.train()
        phanxu.test(args.test_data, args.output)
    elif args.task == 'train':
        phanxu = PhanXu(model_prefix=args.model_prefix, train_files=args.train_data, gold_file_path=args.gold_data)
        phanxu.train()
    elif args.task == 'test':
        phanxu = PhanXu(model_prefix=args.model_prefix)
        phanxu.load_model()
        phanxu.test(args.test_data, args.output)
    else:
        print('Wtf r u saying? all, train or test. That\'s all')
