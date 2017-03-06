from collections import deque
from common import utils


class Configuration:

    def __init__(self, init_buffer=None):
        self.stack = deque([0])                 # Init with ROOT
        self.buffer = deque(init_buffer)
        self.arcs = []

    def __repr__(self):
        return 'S: %s --- B: %s --- A: %s' % (str(self.stack), str(self.buffer), str(self.arcs))

    def op_left_arc(self, label=''):
        if len(self.stack) < 2:
            utils.logger.error("Stack len < 2")
            return

        head = self.stack.pop()
        dep = self.stack.pop()
        self.stack.append(head)                 # Do not remove head
        self.arcs.append((head, dep, label))

    def op_right_arc(self, label=''):
        if len(self.stack) < 2:
            utils.logger.error("Stack len < 2")
            return

        dep = self.stack.pop()
        head = self.stack[-1]                   # Do not remove head
        self.arcs.append((head, dep, label))

    def op_shift(self):
        # if len(self.buffer) > 0:
        #     self.stack.append(self.buffer.popleft())
        # else:
        #     utils.logger.error("Buffer is empty")
        self.stack.append(self.buffer.popleft())

    def is_final(self):
        return len(self.buffer) <= 0 and len(self.stack) < 2

    def get_stack_tops(self):
        if len(self.stack) < 2:
            return -1, -1
        return self.stack[-1], self.stack[-2]

    def is_done(self, words):
        """ If the word in words is done with our transitions"""
        for w in words:
            if w in self.stack or w in self.buffer:
                return False

        return True
