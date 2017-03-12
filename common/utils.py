import logging.config
import os

# Paths
PROJECT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# Logger
logging.config.fileConfig(os.path.join(PROJECT_PATH, 'config/logging.conf'))
logger = logging.getLogger('common')
