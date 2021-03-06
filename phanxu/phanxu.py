import argparse
import io
from collections import Counter
from os.path import exists
import pickle

import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Input
from keras.engine import Model
from keras.layers import Embedding, TimeDistributed, Dense, Lambda, Dropout, BatchNormalization, concatenate, \
    Bidirectional, recurrent
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

from config import *
from common import conllu
from common import utils


class PhanXu:

    def __init__(self, model_prefix, model_type, sen_type,
                 train_files=None, gold_file_path=None, embedding_file_path=None):
        self.train_data = None
        self.train_labels = None
        self.VOCAB_SIZE = 0
        self.word_embedding_matrix = None
        self.model = None
        self.word_index = None
        self.model_prefix = model_prefix
        self.model_type = model_type
        self.sen_type = sen_type
        if train_files is not None and gold_file_path is not None:
            self.read_files(train_files, gold_file_path, embedding_file_path)

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

    def parse_data_2(self, train_raws):
        res_1 = []
        res_2 = []
        for (_, sen1), (_, sen2) in zip(train_raws[0], train_raws[1]):
            ids_1 = sorted([_id for _id in sen1 if _id != '0' and '-' not in _id and '.' not in _id],
                           key=utils.get_id_key)
            ids_2 = sorted([_id for _id in sen2 if _id != '0' and '-' not in _id and '.' not in _id],
                           key=utils.get_id_key)
            res_1.append([self.word_index.get(sen1[_id][conllu.FORM], len(self.word_index)) for _id in ids_1]
                         + [self.word_index.get(sen1[sen1[_id][conllu.HEAD]][conllu.FORM], len(self.word_index)) for _id in ids_1])
            res_2.append([self.word_index.get(sen2[_id][conllu.FORM], len(self.word_index)) for _id in ids_2]
                         + [self.word_index.get(sen2[sen2[_id][conllu.HEAD]][conllu.FORM], len(self.word_index)) for _id in ids_2])

        return [pad_sequences(res_1, maxlen=MAX_LEN), pad_sequences(res_2, maxlen=MAX_LEN)]

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

            if self.model_type == 'compare':
                train_raws = [conllu.CoNLLU(train_file).get_content() for train_file in train_files]
                self.train_data = self.parse_data_2(train_raws)
                self.train_labels = (np.array(self.score(train_raws[0], gold_raw.get_content()))
                                     >= np.array(self.score(train_raws[1], gold_raw.get_content()))).astype(float)
            else:
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
            print('Init word embedding matrix with pre-trained')
            self.word_embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, EMBEDDING_SIZE))

            with io.open(embedding_file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f):
                    values = line.strip().split(' ')
                    word = values[0]
                    if word in self.word_index:
                        try:
                            self.word_embedding_matrix[self.word_index[word]] = np.asarray(values[1:], dtype='float32')
                        except Exception as e:
                            print('Embedding error on line %d' % line_no)

            np.save(self.model_prefix + '_word_embedding.npy', self.word_embedding_matrix)

    def get_model(self, max_size=EMBEDDING_SIZE):
        # Defining model
        print('Define model')
        q1 = Input(shape=(MAX_LEN,), dtype='int32')
        q2 = Input(shape=(MAX_LEN,), dtype='int32')

        if max_size < EMBEDDING_SIZE:
            print('Reduce EMBEDDING_SIZE to %d due to pretrained embedding' % max_size)

        if max_size < SENT_HIDDEN_SIZE:
            print('Reduce SENT_HIDDEN_SIZE to %d due to pretrained embedding' % max_size)

        if self.word_embedding_matrix is None:
            embed = Embedding(self.VOCAB_SIZE + 1, min(max_size, EMBEDDING_SIZE),
                              input_length=MAX_LEN, trainable=TRAIN_EMBED)
        else:
            embed = Embedding(self.VOCAB_SIZE + 1, min(max_size, EMBEDDING_SIZE), weights=[self.word_embedding_matrix],
                              input_length=MAX_LEN, trainable=TRAIN_EMBED)
        embed_q1 = embed(q1)
        embed_q2 = embed(q2)

        translate = TimeDistributed(Dense(min(max_size, SENT_HIDDEN_SIZE), activation=ACTIVATION))

        sent_q1 = translate(embed_q1)
        sent_q2 = translate(embed_q2)

        if self.sen_type == 'lstm':
            sent_embed = Bidirectional(recurrent.LSTM(units=min(max_size, SENT_HIDDEN_SIZE),
                                                      recurrent_dropout=DROPOUT_RATE, dropout=DROPOUT_RATE,
                                                      return_sequences=False))
        else:
            sent_embed = Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=(min(max_size, SENT_HIDDEN_SIZE),))

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
        if self.model_type == 'compare':
            model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
        else:
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

    def test_score(self, input_files, output_file, gold_test_file):
        datas = []
        preds = []
        _score = []
        gold_data = conllu.CoNLLU(gold_test_file).get_content()
        for input_file in input_files:
            datas.append(conllu.CoNLLU(input_file).get_content())
            data = self.parse_data(datas[-1])
            labels = self.score(datas[-1], gold_data)
            _score.append(np.array(self.model.evaluate(data, labels)))
            print("%s: %s" % (input_file, str(_score[-1])))
            preds.append(self.model.predict(data).flatten())

        print("Total: %s" % str(sum(_score)/len(_score)))
        preds = np.array(preds).transpose()
        res = conllu.CoNLLU()
        for _id, row in enumerate(preds):
            print(row)
            pred_max = np.argmax(row)
            res._content.append(datas[pred_max][_id])

        res.to_file(output_file)

    def test_compare(self, input_files, output_file, gold_test_file):
        datas = [conllu.CoNLLU(input_file).get_content() for input_file in input_files]
        gold_data = conllu.CoNLLU(gold_test_file).get_content()
        data = self.parse_data_2(datas)
        labels = (np.array(self.score(datas[0], gold_data))
                  >= np.array(self.score(datas[1], gold_data))).astype(float)
        print(self.model.evaluate(data, labels))
        preds = self.model.predict(data).flatten()

        res = conllu.CoNLLU()
        for _id, row in enumerate(preds):
            pred_max = 0 if row > 0.5 else 1
            res._content.append(datas[pred_max][_id])

        res.to_file(output_file)

    def test(self, input_files, output_file, gold_test_file):
        if self.model_type == 'compare':
            self.test_compare(input_files, output_file, gold_test_file)
        else:
            self.test_score(input_files, output_file, gold_test_file)

    def load_model(self):
        self.model = self.get_model()
        print('Load weights from file')
        self.model.load_weights(self.model_prefix + '.h5')

        print('Load word_index from file')
        self.word_index = pickle.load(open(self.model_prefix + '_word_index.pkl', 'rb'))

        if exists(self.model_prefix + '_word_embedding.npy'):
            print('Load word embedding matrix from file')
            self.word_embedding_matrix = np.load(self.model_prefix + '_word_embedding.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("-m", "--model-prefix")
    parser.add_argument("-i", "--train-data", action='append')
    parser.add_argument("-g", "--gold-data")
    parser.add_argument("-t", "--test-data", action='append')
    parser.add_argument("-d", "--gold-test-data")
    parser.add_argument("-o", "--output")
    parser.add_argument("-e", "--embedding")
    parser.add_argument("-c", "--model-type", default='compare')
    parser.add_argument("-s", "--sentence-type", default='lstm')
    args = parser.parse_args()

    if args.task == 'all':
        phanxu = PhanXu(args.model_prefix, args.model_type, args.sentence_type, args.train_data, args.gold_data, args.embedding)
        phanxu.train()
        phanxu.test(args.test_data, args.output, args.gold_test_data)
    elif args.task == 'train':
        phanxu = PhanXu(args.model_prefix, args.model_type, args.sentence_type, args.train_data, args.gold_data, args.embedding)
        phanxu.train()
    elif args.task == 'test':
        phanxu = PhanXu(args.model_prefix, args.model_type, args.sentence_type)
        phanxu.load_model()
        phanxu.test(args.test_data, args.output, args.gold_test_data)
    else:
        print('Wtf r u saying? all, train or test. That\'s all')
