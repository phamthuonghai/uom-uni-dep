import logging
import logging.config
import os

# Paths
PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

# Logger
logging.config.fileConfig(os.path.join(PROJECT_PATH, 'config/logging.conf'))
logger = logging.getLogger('common')
