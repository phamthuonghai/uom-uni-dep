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
    return [int(t) for t in re.split('[.-]', _id)]
