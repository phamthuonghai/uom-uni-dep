import pickle

from . import utils

# Constants for the column indices
COL_COUNT = 10
ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC = range(COL_COUNT)
COL_NAMES = u"ID,FORM,LEMMA,UPOSTAG,XPOSTAG,FEATS,HEAD,DEPREL,DEPS,MISC".split(u",")
DUMMY_HEAD = {'0': ['0', '', '', '', '', '', -1, '', '', '']}


class CoNLLU:
    """ CoNLL-U format object """

    def __init__(self, file_path=None):
        self._content = []
        if file_path:
            self.from_file(file_path)

    def from_file(self, file_path):
        """ Parse data from CoNLL-U format file """
        with open(file_path, 'r', encoding='utf-8') as f:
            sent_id = ''
            tmp_cont = DUMMY_HEAD.copy()

            for line in f:
                line = line.strip()

                if len(line) == 0:
                    if len(tmp_cont) > 1:
                        self._content.append((sent_id, tmp_cont))
                        tmp_cont = DUMMY_HEAD.copy()

                elif line[0].isdigit():
                    data_line = line.split('\t')

                    if len(data_line) != COL_COUNT:
                        utils.logger.error('Missing data: %s' % line)
                        continue

                    tmp_cont[data_line[ID]] = data_line

                else:
                    data_line = line.split()
                    if len(data_line) == 4 and data_line[1] == 'sent_id':
                        sent_id = data_line[-1]

        if len(tmp_cont) > 1:
            self._content.append((sent_id, tmp_cont))

    def remove_dep(self):
        for _, sentence in self._content:
            for __, w in sentence.items():
                w[HEAD] = ''
                w[DEPREL] = ''
                w[DEPS] = ''

    def to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence in self._content:
                for _id in sorted(sentence.keys(), key=utils.get_id_key)[1:]:
                    f.write('\t'.join([str(x) for x in sentence[_id]]) + '\n')
                f.write('\n')

    def load(self, file_path):
        with open(file_path, 'rb') as fi:
            self._content = pickle.load(fi)

    def save(self, file_path):
        with open(file_path, 'wb') as fo:
            pickle.dump(self._content, fo)

    def get_content(self):
        return self._content

    def set_content(self, _content):
        self._content = _content
