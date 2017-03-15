import argparse

from tqdm import tqdm

from common.conllu import *
from common import utils
from chen_parser.oracle import Oracle
from chen_parser.feature import FeatureExtractor
from chen_parser.arc_standard import Configuration

from chen_parser import settings as c_stt


class Parser:
    def __init__(self, oracle_path_prefix, template_file_path):
        utils.logger.info('Initiating Chen parser')
        self.feature_extractor = FeatureExtractor(template_file_path)
        utils.logger.info('Loading Oracle from file')
        self.oracle = Oracle()
        self.oracle.load(oracle_path_prefix)

    def parse_sentence(self, sentence):
        cur_config = Configuration(sentence.keys())

        signal = 1
        while not cur_config.is_final():
            trans = self.oracle.predict(self.feature_extractor.extract_features(cur_config, sentence),
                                        cur_config)

            if trans[0] == 's':     # shift
                h = cur_config.op_shift()
                if h is None:
                    signal = -1
                    break
            elif trans[0] in 'lr':  # arc [lr]_*
                h, d, l = cur_config.op_arc(trans[0], trans[2:])
                if h is None:
                    signal = -1
                    break
                if c_stt.UPDATE_HEAD:
                    sentence[d][HEAD] = h
                    sentence[d][DEPREL] = l
            else:
                utils.logger.warning('Unknown prediction from Oracle %s' % trans)
                signal = -1
                break

        for arc in cur_config.arcs:
            sentence[arc[1]][HEAD] = arc[0]
            sentence[arc[1]][DEPREL] = arc[2]

        return signal, sentence

    def parse_sentences(self, sentences):
        res = []
        cnt_err = 0
        for _, sentence in tqdm(sentences):
            tmp_sig, tmp_res = self.parse_sentence(sentence)
            if tmp_sig < 0:
                cnt_err += 1

            res.append(tmp_res)

        utils.logger.info('%d fully parsed, %d partially parsed' % (len(sentences) - cnt_err, cnt_err))
        return res

    def parse_conllu_file(self, file_path):
        utils.logger.info('Parsing from file %s ' % file_path)
        conllu_file = CoNLLU(file_path)

        # This is to removed parsed results for testing
        conllu_file.remove_dep()

        return self.parse_sentences(conllu_file.get_content())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file (conllu)")
    parser.add_argument("model", help="output model files (path prefix)")
    parser.add_argument("output", help="output file (conllu)")
    parser.add_argument("-t", "--template", help="template file path",
                        default='./config/chen.template')

    args = parser.parse_args()

    parser = Parser(args.model, args.template)
    tmp = parser.parse_conllu_file(args.input)

    file_out = CoNLLU()
    file_out.set_content(tmp)
    file_out.to_file(args.output)
