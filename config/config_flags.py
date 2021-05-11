''' Config file with constants '''

import os.path as osp

EMBEDDINGS_ROOT = osp.expanduser("../embeddings")

EMBEDDINGS_DATASET_NAME = 'tieredImageNet'

EMBEDDINGS_CROP = 'center'

EMBEDDINGS_FILENAME_TEMPLATE = '_embeddings.pkl'

TRAINING_NUM_OF_EXAMPLES_PER_CLASS = 5 #k in k shot, i.e. Number of training examples in each nth class

NUM_OF_CLASSES = 5 # n in n way

TEST_VALIDATION_NUM_OF_EXAMPLES_PER_CLASS = 599 # original: 600 - TRAINING_NUM_OF_EXAMPLES_PER_CLASS

VALIDATION_NUM_OF_EXAMPLES_PER_CLASS = 15 #Number of examples in each nth class

CHECKPOINT_ROOT = "checkpoint/"

SAVE_ROOT = "checkpoint/save"

TRAIN_SAMPLE_SIZE = 500 #original : 30000

TEST_SAMPLE_SIZE = 10 #original: 100

CHECKPOINT_STEPS = 500 # checkpoint_steps

TOTAL_STEPS = 4000 # num_steps_limit

TOP_M = 5 #original: 1% of train_sample_size (30000)

BATCH_SIZE = 30
