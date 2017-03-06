import os

from common.conllu import *
from chen_parser.arc_standard import Configuration
from common import utils


class FeatureExtractor:
    """ Feature Extractor """
    TEMPLATE_TO_CONLLU = {'w': LEMMA, 't': UPOSTAG, 'l': DEPREL}

    def __init__(self, template_file_path=None):
        self.train_data = []
        self.template = []
        if template_file_path:
            self.template_from_file(template_file_path)

    def template_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    tmp = []
                    p_line = line.strip().split()
                    for t in p_line[:-1]:
                        q = t.split('.')
                        q[-1] = int(q[-1])
                        tmp.append(q)
                    tmp.append((p_line[-1],))
                    self.template.append(tmp)
        except Exception as e:
            utils.logger.error('template_from_file', e)
            raise e

    def get_child(self, sen, id, pos):
        children = sorted([w[ID] for w in sen if w[HEAD] == id])
        return children[pos] if 0 <= pos < len(children) else None

    def extract_features(self, conf, sen):
        res_f = []
        for feature in self.template:
            vl = None
            for cmd in feature:
                if cmd[0] == 's':
                    vl = conf.get_stack(cmd[1])
                elif cmd[0] == 'b':
                    vl = conf.get_buffer(cmd[1])
                elif cmd[0] == 'lc':
                    vl = self.get_child(sen, vl, cmd[1]-1)
                elif cmd[0] == 'rc':
                    vl = self.get_child(sen, vl, -cmd[1])
                elif cmd[0] in self.TEMPLATE_TO_CONLLU:
                    res_f.append(sen[vl][self.TEMPLATE_TO_CONLLU[cmd[0]]])
                    break
                else:
                    utils.logger.error('Unknown command in template %s' % cmd[0])
                if vl is None:
                    res_f.append('')
                    break
        return res_f

    def get_feature_from_sentence(self, sentence):
        cur_config = Configuration([w[ID] for w in sentence[1:]])
        # ls_config = [copy.deepcopy(cur_config)]
        ls_features = []

        while not cur_config.is_final():
            try:
                cur_features = self.extract_features(cur_config, sentence)

                t1, t2 = cur_config.get_stack_tops()
                if t2 is None or t1 is None:                                                        # Stack empty
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

    def feature_from_file(self, file_path):
        conllu_data = CoNLLU()
        conllu_data.from_file(file_path)

        cnt_err = 0
        cnt_success = 0
        for _, sen in conllu_data.get_content().items():
            parsed_sentence = self.get_feature_from_sentence(sen)
            if parsed_sentence:
                self.train_data.append(parsed_sentence)
                cnt_success += 1
            else:
                cnt_err += 1

        utils.logger.info('Parsed %d successes, %d failures' % (cnt_success, cnt_err))

if __name__ == '__main__':
    f_ex = FeatureExtractor(os.path.join(utils.PROJECT_PATH, 'config/chen.template'))
    f_ex.feature_from_file(os.path.join(utils.PROJECT_PATH,
                                        'data/ud-treebanks-conll2017/UD_English/en-ud-dev.conllu'))
