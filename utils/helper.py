try:
    import numpy as np
    from pathlib import Path
    import pickle
    import os
    import os.path as osp
    import sys
    import importlib
    importlib.reload(Config)
except NameError: # It hasn't been imported yet
        import config.config_flags as Config

def get_alpha_weights_path():
    train_path = get_task_dataset_path("train")
    test_path = get_task_dataset_path("test")
    train_name = osp.basename(train_path)
    test_name = osp.basename(test_path)
    weight_name = train_name + "-" + test_name

    checkpoint_path = Config.CHECKPOINT_ROOT
    alpha_weight_path = osp.join(checkpoint_path, weight_name)
    return alpha_weight_path

def get_task_dataset_path(dataset_type):
    db_title = Config.EMBEDDINGS_DATASET_NAME
    checkpoint_path = Config.CHECKPOINT_ROOT
    path = None
    if dataset_type == 'train':
        train_sample_size = Config.TRAIN_SAMPLE_SIZE
        train_tr_size = Config.TRAINING_NUM_OF_EXAMPLES_PER_CLASS
        train_val_size = Config.VALIDATION_NUM_OF_EXAMPLES_PER_CLASS
        path = osp.join(checkpoint_path, "%s_%s_%i_%i_%i" % ("train", db_title, train_sample_size, train_tr_size, train_val_size))
    elif dataset_type == 'test':
        test_sample_size =Config.TEST_SAMPLE_SIZE
        test_tr_size = Config.TRAINING_NUM_OF_EXAMPLES_PER_CLASS
        test_val_size = Config.TEST_VALIDATION_NUM_OF_EXAMPLES_PER_CLASS
        path = osp.join(checkpoint_path, "%s_%s_%i_%i_%i" % ("test", db_title, test_sample_size, test_tr_size, test_val_size))
    return path