import os
import copy

from common.conllu import *
from chen_parser.arc_standard import Configuration
from common import utils


class ChenTrainer:
    def __init__(self):
        self.train_data = []
        return

    def gen_train_data_from_sentence(self, sentence):
        cur_config = Configuration([w[ID] for w in sentence[1:]])
        ls_config = [copy.deepcopy(cur_config)]

        while not cur_config.is_final():
            try:
                t1, t2 = cur_config.get_stack_tops()
                if t2 < 0:                                                                  # Stack empty
                    cur_config.op_shift()
                elif sentence[t2][HEAD] == sentence[t1][ID]:                                # Possible LEFT_ARC
                    cur_config.op_left_arc(sentence[t2][DEPREL])
                elif (sentence[t1][HEAD] == sentence[t2][ID] and                            # Possible RIGHT_ARC
                          cur_config.is_done([w[ID] for w in sentence if w[HEAD] == t1])):  # t2 is done
                    cur_config.op_right_arc(sentence[t1][DEPREL])
                else:
                    cur_config.op_shift()

                ls_config.append(copy.deepcopy(cur_config))
            except Exception as e:
                # print(cur_config)
                # print('\n'.join([str(w) for w in sentence]))
                # print('\n'.join([str(cfg) for cfg in ls_config]))
                # print(t1, t2)
                # print(e)
                return None

        print('\n'.join([str(w) for w in sentence]))
        print('\n'.join([str(cfg) for cfg in ls_config]))
        return ls_config

    def gen_train_data(self, sentences):
        cnt_err = 0
        cnt_success = 0
        for id, sen in sentences.items():
            parsed_sentence = self.gen_train_data_from_sentence(sen)
            if parsed_sentence:
                self.train_data.append(parsed_sentence)
                cnt_success += 1

                if cnt_success > 10:
                    break
            else:
                cnt_err += 1

        utils.logger.info('Parsed %d successes, %d failures' % (cnt_success, cnt_err))

if __name__ == '__main__':
    conllu_data = CoNLLU()
    conllu_data.from_file(os.path.join(utils.PROJECT_PATH,
                                       './data/ud-treebanks-conll2017/UD_English/en-ud-dev.conllu'))
    trainer = ChenTrainer()
    trainer.gen_train_data(conllu_data.get_content())

    print(trainer.train_data[:10])
