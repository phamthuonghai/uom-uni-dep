import os

from common.conllu import *
from common import utils
from chen_parser.arc_standard import Configuration
from chen_parser import settings


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
        children = sorted([w[ID] for w in sen if w[HEAD] == _id])
        return children[pos] if 0 <= pos < len(children) else None

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
        cur_config = Configuration([w[ID] for w in sentence[1:]])
        # ls_config = [copy.deepcopy(cur_config)]
        ls_features = []

        while not cur_config.is_final():
            try:
                cur_features = self.extract_features(cur_config, sentence)

                t1, t2 = cur_config.get_stack_tops()
                if t2 is None or t1 is None:                                                # Stack empty
                    cur_config.op_shift()
                    l_op = 'shift'
                elif sentence[t2][HEAD] == sentence[t1][ID]:                                # Possible LEFT_ARC
                    cur_config.op_left_arc(sentence[t2][DEPREL])
                    l_op = 'l_' + sentence[t2][DEPREL]
                elif (sentence[t1][HEAD] == sentence[t2][ID] and                            # Possible RIGHT_ARC
                          cur_config.is_done([w[ID] for w in sentence if w[HEAD] == t1])):  # t2 is done
                    cur_config.op_right_arc(sentence[t1][DEPREL])
                    l_op = 'r_' + sentence[t1][DEPREL]
                else:
                    cur_config.op_shift()
                    l_op = 'shift'

                # ls_config.append(copy.deepcopy(cur_config))
                ls_features.append((cur_features, l_op))
            except Exception as e:
                # print(cur_config)
                # print('\n'.join([str(w) for w in sentence]))
                # print('\n'.join([str(cfg) for cfg in ls_config]))
                # print(t1, t2)
                utils.logger.debug(e)
                return None

        # print('\n'.join([str(w) for w in sentence]))
        # print('\n'.join([str(cfg) for cfg in ls_features]))
        return ls_features

    def feature_from_parsed_file(self, file_path):
        conllu_data = CoNLLU(file_path)
        self.list_feature_label = []

        cnt_err = 0
        cnt_success = 0
        for _, sen in conllu_data.get_content():
            parsed_sentence = self.get_feature_parsed_sentence(sen)
            if parsed_sentence is not None:
                self.list_feature_label.append(parsed_sentence)
                cnt_success += 1
            else:
                cnt_err += 1

        utils.logger.info('Parsed %d successes, %d failures' % (cnt_success, cnt_err))

    def save(self, file_path):
        with open(file_path, 'wb') as fo:
            pickle.dump(self.list_feature_label, fo)

if __name__ == '__main__':
    f_ex = FeatureExtractor(os.path.join(utils.PROJECT_PATH, 'config/chen.template'))
    f_ex.feature_from_parsed_file(os.path.join(utils.PROJECT_PATH,
                                               'data/ud-treebanks-conll2017/UD_English/en-ud-dev.conllu'))
    f_ex.save(os.path.join(utils.PROJECT_PATH, 'models/en-ud-dev.pkl'))