import os

from tqdm import tqdm

from common.conllu import *
from common import utils
from chen_parser.oracle import Oracle
from chen_parser.feature import FeatureExtractor
from chen_parser.arc_standard import Configuration


class Parser:
    def __init__(self, oracle_path_prefix, template_file_path):
        utils.logger.info('Initiating Chen parser')
        self.feature_extractor = FeatureExtractor(template_file_path)
        utils.logger.info('Loading Oracle from file')
        self.oracle = Oracle()
        self.oracle.load(oracle_path_prefix)

    def parse_sentence(self, sentence):
        cur_config = Configuration(sentence.keys())

        while not cur_config.is_final():
            try:
                trans = self.oracle.predict(self.feature_extractor.extract_features(cur_config, sentence))

                if trans[0] == 's':     # shift
                    cur_config.op_shift()
                elif trans[0] == 'l':   # left arc l_*
                    cur_config.op_left_arc(trans[2:])
                elif trans[0] == 'r':   # right arc r_*
                    cur_config.op_right_arc(trans[2:])
            except Exception as e:
                utils.logger.debug(e)
                return None

        for arc in cur_config.arcs:
            sentence[arc[1]][HEAD] = arc[0]
            sentence[arc[1]][DEPREL] = arc[2]

        return sentence

    def parse_sentences(self, sentences):
        res = []
        cnt_err = 0
        for _, sentence in tqdm(sentences):
            tmp_res = self.parse_sentence(sentence)
            if tmp_res is None:
                cnt_err += 1
                res.append(sentence)
            else:
                res.append(tmp_res)

        utils.logger.info('Parsed %d successes, %d failures' % (len(sentences) - cnt_err, cnt_err))
        return res

    def parse_conllu_file(self, file_path):
        utils.logger.info('Parsing from file %s ' % file_path)
        conllu_file = CoNLLU(file_path)

        # This is to removed parsed results for testing
        conllu_file.remove_dep()

        return self.parse_sentences(conllu_file.get_content())


if __name__ == '__main__':
    parser = Parser(os.path.join(utils.PROJECT_PATH, 'models/en-ud-dev'),
                    os.path.join(utils.PROJECT_PATH, 'config/chen.template'))
    tmp = parser.parse_conllu_file(os.path.join(utils.PROJECT_PATH,
                                                'data/ud-treebanks-conll2017/UD_English/en-ud-dev.conllu'))

    file_out = CoNLLU()
    file_out.set_content(tmp)
    file_out.to_file(os.path.join(utils.PROJECT_PATH,
                                  'data/ud-treebanks-conll2017/UD_English/en-ud-res.conllu'))
