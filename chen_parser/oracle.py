import os
import pickle
from collections import Counter

import numpy as np
from keras.engine import Input, Model, merge
from keras.layers import Embedding, Flatten, Dense
from keras.optimizers import Adagrad
from keras.utils.np_utils import to_categorical

from chen_parser import settings as c_stt
from common import utils


class Oracle:
    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.trn_encoder = {}
        self.trn_decoder = {}

        self.model = None

    def train_model_from_file(self, file_path):
        utils.logger.info('Loading training set from file')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            utils.logger.info('Retrieving dictionaries for tokens in features')
            tokens = {k: [] for k in c_stt.DATA_TYPES}
            transitions = set()
            for sentence_steps in data:
                for (features, target) in sentence_steps:
                    for _dt, _tokens in features.items():
                        tokens[_dt] += _tokens
                    transitions.add(target)

            unk_id = {}
            for _dt in c_stt.DATA_TYPES:
                freqs = Counter(tokens[_dt])
                vocab = sorted(freqs.keys(), key=freqs.get, reverse=True)
                if c_stt.VOCAB_LIMIT[_dt] > 0:
                    vocab = vocab[:c_stt.VOCAB_LIMIT[_dt]]

                self.encoder[_dt] = {word: i for (i, word) in enumerate(vocab)}
                self.decoder[_dt] = {i: word for (i, word) in enumerate(vocab)}
                unk_id[_dt] = len(vocab)

            self.trn_encoder = {transition: i for (i, transition) in enumerate(sorted(transitions))}
            self.trn_decoder = {i: transition for (i, transition) in enumerate(sorted(transitions))}

            utils.logger.info('Mapping tokens to ids in dictionaries')
            trainset = {k: [] for k in c_stt.DATA_TYPES}
            trainset_transitions = list()
            for sentence_steps in data:
                for (features, target) in sentence_steps:
                    for _dt, _tokens in features.items():
                        trainset[_dt].append([self.encoder[_dt].get(_tk, unk_id[_dt]) for _tk in _tokens])
                    trainset_transitions.append(self.trn_encoder[target])

            trainset_mat = [np.array(trainset[k], 'int16') for k in c_stt.DATA_TYPES]
            trainset_transitions_mat = to_categorical(trainset_transitions, nb_classes=None)
            # np.zeros((len(trainset_transitions), len(self.trn_encoder)), 'bool')
            # trainset_transitions_mat[np.arange(len(trainset_transitions)), trainset_transitions] = 1

            utils.logger.info('Defining model')
            input_layers = {}
            embedding_layers = {}
            for _dt in c_stt.DATA_TYPES:
                input_layers[_dt] = Input(shape=(len(trainset[_dt][0]),), dtype='int32')
                embedding_layers[_dt] = Flatten()(
                                            Embedding(input_dim=unk_id[_dt]+1,
                                                      output_dim=c_stt.EMBEDDING_SIZE)(
                                                input_layers[_dt]))

            main_input_layer = merge([embedding_layers[_dt] for _dt in c_stt.DATA_TYPES],
                                     mode='concat', concat_axis=1)
            result_layer = Dense(len(self.trn_encoder), activation='softmax')(
                                Dense(c_stt.HIDDEN_LAYER_SIZE, activation=lambda x: x**3)(
                                    main_input_layer))

            self.model = Model(input=[input_layers[_dt] for _dt in c_stt.DATA_TYPES], output=result_layer)

            utils.logger.info('Compiling model')
            self.model.compile(optimizer=Adagrad(), loss='categorical_crossentropy')

            utils.logger.info('Training model')
            self.model.fit(trainset_mat, trainset_transitions_mat,
                           batch_size=c_stt.BATCH_SIZE, nb_epoch=c_stt.N_EPOCH)

    def save(self, path_prefix):
        utils.logger.info('Saving oracle to files:')
        utils.logger.info(' - ' + path_prefix + '-dict.pkl')
        with open(path_prefix + '-dict.pkl', 'wb') as fo:
            pickle.dump((self.encoder, self.decoder, self.trn_encoder, self.trn_decoder), fo)

        utils.logger.info(' - ' + path_prefix + '-model.h5')
        self.model.save(path_prefix + '-model.h5')

    def load(self, path_prefix):
        utils.logger.info('Loading oracle from files:')
        utils.logger.info(' - ' + path_prefix + '-dict.pkl')
        with open(path_prefix + '-dict.pkl', 'rb') as fi:
            pickle.load(fi, (self.encoder, self.decoder, self.trn_encoder, self.trn_decoder))

        utils.logger.info(' - ' + path_prefix + '-model.h5')
        self.model.load_model(path_prefix + '-model.h5')

if __name__ == '__main__':
    oracle = Oracle()
    oracle.train_model_from_file(os.path.join(utils.PROJECT_PATH, 'models/en-ud-dev.pkl'))
    oracle.save(os.path.join(utils.PROJECT_PATH, 'models/en-ud-dev'))
