import os
import re
import logging.config

# Paths
PROJECT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# Logger
logging.config.fileConfig(os.path.join(PROJECT_PATH, 'config/logging.conf'))
logger = logging.getLogger('common')


def get_id_key(_id):
    """ Return a list of id component for ID sorting """
    t = [int(t) for t in re.split('[.-]', _id)]
    if len(t) == 1:
        t.append(0)     # single entry
    elif '-' in _id:
        t = [t[0], -1]   # compound get on top of its components
    return t
