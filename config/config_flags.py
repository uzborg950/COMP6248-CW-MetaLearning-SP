''' Config file with constants '''

import os.path as osp

EMBEDDINGS_ROOT = osp.expanduser("../embeddings")

EMBEDDINGS_DATASET_NAME = 'tieredImageNet'

EMBEDDINGS_CROP = 'center'

EMBEDDINGS_FILENAME_TEMPLATE = '_embeddings.pkl'

TRAINING_NUM_OF_EXAMPLES = 1 #k in k shot, i.e. Number of training examples in each nth class

NUM_OF_CLASSES = 5 # n in n way

VALIDATION_NUM_OF_EXAMPLES = 15 #Number of examples in each nth class

CHECKPOINT_ROOT = "checkpoint/"

SAVE_ROOT = "checkpoint/save"