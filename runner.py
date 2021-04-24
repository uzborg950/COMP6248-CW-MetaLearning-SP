#This script is the main driver :
#1. Set ups metalearning tasks and persists in filesystem
#2. Warm start parameters using unconditional meta learning
#3. running structured prediction (metatraining + metatesting)


try:
    import torch
    import numpy as np
    from pathlib import Path
    import pickle
    import os
    import os.path as osp
    import importlib
    importlib.reload(dp)
    importlib.reload(Config)
except NameError: # It hasn't been imported yet
        import data_load.data_provider as dp
        import config.config_flags as Config



def run_lsmeta_unlimited(lam, l2_weight, layers, num=None):
    db_title = Config.EMBEDDINGS_DATASET_NAME
    tr_size = Config.TRAINING_NUM_OF_EXAMPLES_PER_CLASS
    layer_str = "_".join(map(str, layers))
    save_path = osp.abspath(osp.join(chkpoint_root, "lsmeta%s_%i_%f_%f_%s" % (db_title, tr_size, lam, l2_weight, layer_str)))
    if num is not None: #To keep multiple instances
        save_path += "_%i" % num
    run_training_loop(save_path, lam, l2_weight, layers)

def main(argv):
    os.makedirs(Config.SAVE_ROOT, exist_ok=True)
    if argv[1] == "gen_db":
        print("Generating tasks and compute their alpha weights")
        populate_db()
    elif argv[1] == "uncon_meta":
        print("Learning warmstart parameters with unconditional meta-learning")
        run_lsmeta_unlimited(0.1, 1e-6, (640, 640))


def populate_db():
    train_sample_size = Config.TRAIN_SAMPLE_SIZE
    test_sample_size =Config.TEST_SAMPLE_SIZE
    train_name = build_db("train", train_sample_size)
    test_name = build_db("test", test_sample_size)

    alpha_path =  compute_alpha(train_name, test_name)
    return alpha_path

def build_db(dataset_type_pkl, sample_size):
    checkpoint_path = Config.CHECKPOINT_ROOT
    db_title = Config.EMBEDDINGS_DATASET_NAME


    num_classes = Config.NUM_OF_CLASSES
    tr_size = Config.TRAINING_NUM_OF_EXAMPLES_PER_CLASS

    if dataset_type_pkl == "test":
        val_size = Config.TEST_VALIDATION_NUM_OF_EXAMPLES_PER_CLASS
    else:
        val_size = Config.VALIDATION_NUM_OF_EXAMPLES_PER_CLASS

    save_path = osp.join(checkpoint_path, "%s_%s_%i_%i_%i" % (dataset_type_pkl, db_title, sample_size, tr_size, val_size))

    if not osp.exists(save_path):
        provider = dp.DataProvider(dataset_type_pkl, debug=False, verbose=True)
        provider.create_db(sample_size, num_classes, tr_size, val_size)
        provider.save_db(save_path)

    return osp.basename(save_path)

def compute_alpha(train_name, test_name, lam=1e-8):
    checkpoint_path = Config.CHECKPOINT_ROOT
    weight_name = train_name + "-" + test_name
    weight_save = osp.join(checkpoint_path, weight_name)

    if not osp.exists(weight_save):
        test_save = osp.join(checkpoint_path, test_name)
        test_sigs = get_sig_matrix(test_save) #Stacks up signature vectors for all tasks into a matrix

        train_save = osp.join(checkpoint_path, train_name)
        train_sigs = get_sig_matrix(train_save)

        print("train_sigs matrix size:", train_sigs[0].shape)
        
        ''' Calculate v(.) evaluation vector'''
        train_test_dist = list(map(lambda x: max_mean_discrepancy(x[0], x[1]), zip(train_sigs, test_sigs)))

        V = gaussian_kernel(train_test_dist)
        del test_sigs, train_test_dist

        ''' Calculate K '''
        train_K = compute_kernel(train_sigs)
        alpha_weights = np.linalg.inv(train_K + lam * np.identity(train_K.shape[0])) @ V

        with open(weight_save, "wb") as f:
            pickle.dump(alpha_weights, f)
    return weight_save


def get_sig_matrix(db_path):
    db = unpickle(db_path) #loads tasks (task_sig, label_array, path_array) x sample_size
    sigs = [e[0] for e in db] #task_sig
    if isinstance(sigs[0], tuple):
        print("Only one sample", len(sigs))
        sigs = map(lambda x:x.astype(np.float64), map(np.stack, zip(*sigs)))
        return list(sigs) #(640,) I guess?
    else:
        print("Multiple samples (matrix)", len(sigs))
        sigs = np.stack(sigs).astype(np.float64) # matrix size: (sample_size,640)
        return [sigs]

def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def max_mean_discrepancy(Phi_1, Phi_2=None):
    ''' Take L2 norm squared '''
    #We can have multiple training tasks/D_tr. Test/Target/D task will always be one
    #Output should be distance of each row of D_tr against the only row of D 
    if Phi_2 is None:
        Phi_2 = Phi_1
    diff = Phi_1 - Phi_2
    diff = diff ** 2
    dist = np.sum(diff,axis=1)[:,np.newaxis]
    return dist

def gaussian_kernel(distance_list, bandwidth=1.0):
    return np.exp(-distance_list[0]/bandwidth)

def compute_kernel(sigs):
    n = sigs[0].shape[0]
    ret = None
    for i in range(n):
        k_i = max_mean_discrepancy(sigs[0], sigs[0][i,:]) # k_i = 300x1
        if ret is None:
           ret = k_i
        else:
            ret = np.append(ret, k_i, axis=1) # one by one appends a 300x1 vector columnwise 
    return np.exp(-ret)

def their_compute_kernel(sigs, weights, func=None):
    n = sigs[0].shape[0]
    ret = np.zeros([n, n])

    for i in range(len(sigs)):
        if func is None:
            ret += weights[i] * build_dist(sigs[i])
        else:
            ret += func(-weights[i] * build_dist(sigs[i]))
    if func is None:
        return np.exp(-ret)
    else:
        return ret / len(sigs)

def build_dist(feat, feat_2=None):
    if feat_2 is None:
        feat_2 = feat
    dist = np.sum(feat ** 2, axis=1, keepdims=True) + np.sum(feat_2 ** 2, axis=1)[np.newaxis, :] \
           - 2 * np.matmul(feat, feat_2.T)
    return dist