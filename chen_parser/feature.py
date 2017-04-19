import argparse
import copy
import sys

from tqdm import tqdm

from common.conllu import *
from common import utils
from chen_parser.arc_standard import Configuration
from chen_parser import settings
from pprint import pprint

class FeatureExtractor:
    """ Feature Extractor """

    def __init__(self, template_file_path=None):
        self.list_feature_label = []
        self.template = []
        if template_file_path:
            self.template_from_file(template_file_path)

    def template_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if len(line) > 1 and line[0] != '#':
                        p_line = line.strip().split()
                        tmp = [[p_line[0], None]]
                        for t in p_line[1:]:
                            q = t.split('.')
                            q[-1] = int(q[-1])
                            tmp.append(q)
                        self.template.append(tmp)
        except Exception as e:
            utils.logger.error(e)
            raise e

    @staticmethod
    def get_child(sen, _id, pos):
        children = sorted([k for k, w in sen.items() if w[HEAD] == _id], key=utils.get_id_key)
        return children[pos] if -len(children) <= pos < len(children) else None

    def extract_features(self, conf, sen):
        res_f = {k: [] for k in settings.DATA_TYPES}
        for feature in self.template:
            vl = None
            data_type = feature[0][0]
            for cmd in feature[1:]:
                if cmd[0] == 's':
                    vl = conf.get_stack(cmd[1])
                elif cmd[0] == 'b':
                    vl = conf.get_buffer(cmd[1])
                elif cmd[0] == 'lc':
                    vl = self.get_child(sen, vl, cmd[1]-1)
                elif cmd[0] == 'rc':
                    vl = self.get_child(sen, vl, -cmd[1])
                else:
                    utils.logger.error('Unknown command in template %s' % cmd[0])
            if vl is None:
                res_f[data_type].append('')
            else:
                res_f[data_type].append(sen[vl][settings.TEMPLATE_TO_CONLLU[data_type]])
        return res_f

    def get_feature_parsed_sentence(self, sentence):
        cur_config = Configuration(sentence.keys())
        signal = 1
        ls_features = []

        # Create a copy of sentence without HEAD & DEPREL info
        sen_new = copy.deepcopy(sentence)
        for k, v in sen_new.items():
            v[HEAD] = ''
            v[DEPREL] = ''
            v[DEPS] = ''

        while not cur_config.is_final():
            cur_features = self.extract_features(cur_config, sen_new)
            dead_trans = cur_config.dead_trans()

            t1, t2 = cur_config.get_stack_tops()
            if 's' not in dead_trans and t2 is None or t1 is None:                      # Stack empty
                cur_config.op_shift()
                l_op = 'shift'
            elif 'l' not in dead_trans and sentence[t2][HEAD] == sentence[t1][ID]:      # Possible LEFT_ARC
                h, d, l = cur_config.op_arc('l', sentence[t2][DEPREL])
                sen_new[d][HEAD] = h
                sen_new[d][DEPREL] = l
                l_op = 'l_' + sentence[t2][DEPREL]
            elif ('r' not in dead_trans and sentence[t1][HEAD] == sentence[t2][ID] and  # Possible RIGHT_ARC
                    cur_config.is_done([k for k, w in sentence.items() if w[HEAD] == t1])):  # t2 is done
                h, d, l = cur_config.op_arc('r', sentence[t1][DEPREL])
                sen_new[d][HEAD] = h
                sen_new[d][DEPREL] = l
                l_op = 'r_' + sentence[t1][DEPREL]
            elif 's' not in dead_trans:
                cur_config.op_shift()
                l_op = 'shift'
            else:
                # print(cur_config)
                # print('\n'.join([str(w) for _, w in sentence.items()]))
                # print('\n'.join([str(cfg) for cfg in ls_config]))
                # print(t1, t2)
                signal = -1
                break

            ls_features.append((cur_features, l_op))

        # print('\n'.join([str(w) for w in sentence]))
        # print('\n'.join([str(cfg) for cfg in ls_features]))
        return signal, ls_features

    def feature_from_parsed_file(self, file_path, is_path=True):
        conllu_data = CoNLLU(file_path, is_path)
        self.list_feature_label = []

        utils.logger.info('Parsing sentences from %s' % file_path)

        cnt_err = 0
        cnt_success = 0
        for _, sen in tqdm(conllu_data.get_content()):
            tmp_sig, parsed_sentence = self.get_feature_parsed_sentence(sen)
            if tmp_sig > 0:
                cnt_success += 1
            else:
                cnt_err += 1
            self.list_feature_label.append(parsed_sentence)

        utils.logger.info('%d fully parsed, %d partially parsed' % (cnt_success, cnt_err))

    def save(self, file_path, pri):
        with open(file_path, 'wb') as fo:
            pickle.dump((self.list_feature_label, pri), fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', help="input file (conllu)",
                        default=sys.stdin)
    parser.add_argument("-o", "--output", help="output feature file (pickle)")
    parser.add_argument("-t", "--template", help="template file",
                        default='./config/chen.template')
    parser.add_argument("-p", "--primary", 
                        help="number of sents in primary treebank",
                        default='-1')

    args = parser.parse_args()

    f_ex = FeatureExtractor(args.template)
    f_ex.feature_from_parsed_file(args.input, isinstance(args.input, str))
    f_ex.save(args.output, args.primary)
