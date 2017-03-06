import pickle

from . import utils

# Constants for the column indices
COL_COUNT = 10
ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC = range(COL_COUNT)
COL_NAMES = u"ID,FORM,LEMMA,UPOSTAG,XPOSTAG,FEATS,HEAD,DEPREL,DEPS,MISC".split(u",")
DUMMY_HEAD = [0, 'ROOT', 'root', '', '', '', -1, '', '', '']


class CoNLLU:
    """ CoNLL-U format object """

    def __init__(self, file_path=None):
        self._content = {}
        if file_path:
            self.from_file(file_path)

    def from_file(self, file_path):
        """ Parse data from CoNLL-U format file """
        with open(file_path, 'r') as f:
            sent_id = ''
            tmp_cont = [DUMMY_HEAD]

            for line in f:
                line = line.strip()

                if len(line) == 0:
                    self._content[sent_id] = tmp_cont
                    tmp_cont = [DUMMY_HEAD]

                elif line[0].isdigit():
                    data_line = line.split('\t')

                    if len(data_line) != COL_COUNT:
                        utils.logger.error('Missing data: %s' % line)
                        continue

                    # Convert ID and HEAD to number
                    data_line[ID] = int(data_line[ID])
                    data_line[HEAD] = int(data_line[HEAD])
                    tmp_cont.append(data_line)

                else:
                    data_line = line.split()
                    if len(data_line) == 4 and data_line[1] == 'sent_id':
                        sent_id = data_line[-1]

    def load(self, file_path):
        with open(file_path, 'rb') as fi:
            self._content = pickle.load(fi)

    def save(self, file_path):
        with open(file_path, 'wb') as fo:
            pickle.dump(self._content, fo)

    def get_content(self):
        return self._content
