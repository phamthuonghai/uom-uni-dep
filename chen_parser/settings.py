from common.conllu import *

TEMPLATE_TO_CONLLU = {'w': FORM, 't': UPOSTAG, 'l': DEPREL}
VOCAB_LIMIT = {'w': 10000, 't': -1, 'l': -1}
# DATA_TYPES = ['w', 't', 'l']
DATA_TYPES = ['w', 't']
EMBEDDING_SIZE = 50
HIDDEN_LAYER_SIZE = 200
BATCH_SIZE = 200
N_EPOCH = 10
