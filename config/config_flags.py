''' Config file with constants '''

import os.path as osp

EMBEDDINGS_ROOT = osp.expanduser("..\\embeddings")

EMBEDDINGS_DATASET_NAME = 'tieredImageNet'

EMBEDDINGS_CROP = 'center'

EMBEDDINGS_FILENAME_TEMPLATE = '_embeddings.pkl'