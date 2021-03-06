from collections import deque
from common import utils


class Configuration:

    def __init__(self, init_buffer=None):
        self.stack = deque(['0'])                 # Init with ROOT
        if init_buffer is not None:
            # Remove '0' for _ROOT_ and multi-word '00-00' and empty node '0.0'
            init_buffer = sorted([_id for _id in init_buffer if _id != '0' and '-' not in _id and '.' not in _id],
                                 key=utils.get_id_key)
        self.buffer = deque(init_buffer)
        self.arcs = []

    def __repr__(self):
        return '---\nS: %s\nB: %s\nA: %s\n---' % (str(self.stack), str(self.buffer), str(self.arcs))

    def op_arc(self, _dir, label=''):
        if len(self.stack) < 2:
            utils.logger.debug("Stack len < 2")
            return None, None, None

        if _dir == 'l':
            head = self.stack.pop()
            dep = self.stack.pop()
            self.stack.append(head)                 # Do not remove head
        elif _dir == 'r':
            dep = self.stack.pop()
            head = self.stack[-1]                   # Do not remove head
        else:
            return None, None, None
        self.arcs.append((head, dep, label))
        return head, dep, label

    def op_shift(self):
        if len(self.buffer) > 0:
            t = self.buffer.popleft()
            self.stack.append(t)
            return t
        else:
            utils.logger.debug("Buffer is empty")
            return None

    def is_final(self):
        return len(self.buffer) <= 0 and len(self.stack) < 2

    def get_stack(self, _id):
        if len(self.stack) >= _id:
            return self.stack[-_id]
        return None

    def get_buffer(self, _id):
        if len(self.buffer) >= _id:
            return self.buffer[_id-1]
        return None

    def get_stack_tops(self):
        return self.get_stack(1), self.get_stack(2)

    def is_done(self, words):
        """ If all the word in words is done with our transitions """
        for w in words:
            if w in self.stack or w in self.buffer:
                return False

        return True

    def dead_trans(self):
        res = ''
        lb = len(self.buffer)
        ls = len(self.stack)

        if lb < 1:      # Nothing to shift
            res += 's'

        if ls < 2:      # Nothing to pop from stack
            res += 'lr'
        elif ls == 2:   # Only _ROOT_ and root token left
            res += 'l'  # No _ROOT_ <- root relation
            if lb > 0:  # Something still needs to be processed
                res += 'r'
        return res
