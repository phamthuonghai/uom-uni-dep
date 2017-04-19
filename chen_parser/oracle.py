import argparse
import pickle
from collections import Counter

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Input, Model, merge
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adagrad
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

from chen_parser import settings as c_stt
from common import utils


def cubic_activation(x):
    return x ** 3


class Oracle:
    def __init__(self, path_prefix):
        self.encoder = {}
        self.decoder = {}
        self.trn_encoder = {}
        self.trn_decoder = {}
        self.unk_id = {}

        self.path_prefix = path_prefix

        self.model = None

    def train_model_from_file(self, file_path):
        utils.logger.info('Loading training set from file')
        with open(file_path, 'rb') as f:
            data, pri_sents = pickle.load(f)

            utils.logger.info('Retrieving dictionaries for tokens in features')
            tokens = {k: [] for k in c_stt.DATA_TYPES}
            transitions = set()
            for sentence_steps in data:
                for (features, target) in sentence_steps:
                    for _dt, _tokens in features.items():
                        tokens[_dt] += _tokens
                    transitions.add(target)

            self.unk_id = {}
            for _dt in c_stt.DATA_TYPES:
                freqs = Counter(tokens[_dt])
                vocab = sorted(freqs.keys(), key=freqs.get, reverse=True)
                if c_stt.VOCAB_LIMIT[_dt] > 0:
                    vocab = vocab[:c_stt.VOCAB_LIMIT[_dt]]

                self.encoder[_dt] = {word: i for (i, word) in enumerate(vocab)}
                self.decoder[_dt] = {i: word for (i, word) in enumerate(vocab)}
                # NOTE: should some data types not have unknown token?
                self.unk_id[_dt] = len(vocab)

            self.trn_encoder = {transition: i for (i, transition) in enumerate(sorted(transitions))}
            self.trn_decoder = {i: transition for (i, transition) in enumerate(sorted(transitions))}

            utils.logger.info('Mapping tokens to ids in dictionaries')
            trainset = {k: [] for k in c_stt.DATA_TYPES}
            trainset_transitions = list()

            # could be neater
            for sent_no, sentence_steps in enumerate(data):
                if sent_no == int(pri_sents):
                    weight_array = np.full(len(trainset_transitions), 1)
                for (features, target) in sentence_steps:
                    for _dt, _tokens in features.items():
                        trainset[_dt].append([self.encoder[_dt].get(_tk, self.unk_id[_dt]) for _tk in _tokens])
                    trainset_transitions.append(self.trn_encoder[target])

            weight_array = np.append(weight_array,
                    np.full(len(trainset_transitions) - len(weight_array),
                        0.8))
            self.save_dict(self.path_prefix)

            trainset_mat = [np.array(trainset[k], 'int16') for k in c_stt.DATA_TYPES]
            trainset_transitions_mat = to_categorical(trainset_transitions, nb_classes=None)
            # np.zeros((len(trainset_transitions), len(self.trn_encoder)), 'bool')
            # trainset_transitions_mat[np.arange(len(trainset_transitions)), trainset_transitions] = 1

            utils.logger.info('Defining model')
            input_layers = {}
            embedding_layers = {}
            for _dt in c_stt.DATA_TYPES:
                input_layers[_dt] = Input(shape=(len(trainset[_dt][0]),), dtype='int16')
                _embedding_layer = Embedding(input_dim=self.unk_id[_dt]+1,
                                             output_dim=c_stt.EMBEDDING_SIZE)(input_layers[_dt])
                embedding_layers[_dt] = Flatten()(_embedding_layer)

            main_input_layer = merge([embedding_layers[_dt] for _dt in c_stt.DATA_TYPES],
                                     mode='concat', concat_axis=1)
            # TODO: in Keras 2.0.0, W_regularizer -> kernel_regularizer
            hidden_layer = Dense(c_stt.HIDDEN_LAYER_SIZE, activation=cubic_activation,
                                 W_regularizer=l2(c_stt.REG_PARAM))(main_input_layer)
            hd_dropout_layer = Dropout(c_stt.DROP_OUT)(hidden_layer)
            result_layer = Dense(len(self.trn_encoder), activation='softmax')(hd_dropout_layer)

            self.model = Model(input=[input_layers[_dt] for _dt in c_stt.DATA_TYPES], output=result_layer)

            utils.logger.info('Compiling model')
            self.model.compile(optimizer=Adagrad(lr=c_stt.LEARNING_RATE), loss='categorical_crossentropy')

            utils.logger.info('Training model')
            
            callbacks = [ModelCheckpoint(self.path_prefix + '-model.h5', save_best_only=True, save_weights_only=False),
                         EarlyStopping(min_delta=c_stt.STOP_MIN_DELTA, patience=c_stt.STOP_PATIENCE)]
            self.model.fit(trainset_mat, trainset_transitions_mat,
                    sample_weight=weight_array, validation_split=c_stt.VALID_SET,
                           batch_size=c_stt.BATCH_SIZE, nb_epoch=c_stt.N_EPOCH, callbacks=callbacks)

            utils.logger.info('Model saved to ' + self.path_prefix + '-model.h5')

    def predict(self, _feature, _config):
        fts = []
        for _dt in c_stt.DATA_TYPES:
            tmp_ft = np.array([self.encoder[_dt].get(_tk, self.unk_id[_dt]) for _tk in _feature[_dt]], 'int16')
            fts.append(tmp_ft.reshape(1, tmp_ft.shape[0]))

        pred = self.model.predict(fts)
        # filter out impossible transitions
        pred[0, [v for k, v in self.trn_encoder.items() if k[0] in _config.dead_trans()]] = -1

        return self.trn_decoder.get(np.argmax(pred[0]), '')

    def save_dict(self, path_prefix):
        utils.logger.info('Saving dictionary to ' + path_prefix + '-dict.pkl')
        with open(path_prefix + '-dict.pkl', 'wb') as fo:
            pickle.dump((self.encoder, self.decoder, self.trn_encoder, self.trn_decoder, self.unk_id), fo)

    def save_model(self, path_prefix):
        utils.logger.info('Saving model to ' + path_prefix + '-model.h5')
        self.model.save(path_prefix + '-model.h5')

    def load(self):
        utils.logger.info('Loading oracle from files:')
        utils.logger.info(' - ' + self.path_prefix + '-dict.pkl')
        with open(self.path_prefix + '-dict.pkl', 'rb') as fi:
            (self.encoder, self.decoder, self.trn_encoder, self.trn_decoder, self.unk_id) = pickle.load(fi)

        utils.logger.info(' - ' + self.path_prefix + '-model.h5')
        self.model = load_model(self.path_prefix + '-model.h5', custom_objects={'cubic_activation': cubic_activation})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input feature file (pickle)")
    parser.add_argument("output", help="output model files (path prefix)")

    args = parser.parse_args()

    oracle = Oracle(args.output)
    oracle.train_model_from_file(args.input)
