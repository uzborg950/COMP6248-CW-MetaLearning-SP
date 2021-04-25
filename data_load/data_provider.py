''' Loads embeddings and saves checkpoint data (generated task database, training parameters) '''

try:
    import collections
    import numpy as np
    import pickle
    import os
    import os.path as osp
    import torch
    from pathlib import Path
    import importlib
    import random
    importlib.reload(Config)
except NameError: # It hasn't been imported yet
        import config.config_flags as Config

#Dataset type: train, val, test
class DataProvider(object):
    def __init__(self, dataset_type, debug=False, verbose=False):
        self._dataset_type = dataset_type
        self._verbose = verbose
        if(debug == True):
            self._embeddings_data = self._load_embeddings_data(dataset_type)
            self._index_data(self._embeddings_data)
        else:
            self._index_data(self._load_embeddings_data(dataset_type)) #Saves memory
        
    
    # Valid output in debug mode only
    def get_embeddings_data(self):
        return self._embeddings_data
    
    #For test only
    def get_indexed_data(self):
            return self._class_image_file_dict, self._image_file_embeddings_dict
    
    def _load_embeddings_data(self, dataset_type):
        embeddings_path = self._get_embeddings_path(dataset_type)
        embeddings_file =open(embeddings_path,'rb')
        pkl_data = pickle.load(embeddings_file, encoding="latin1")
        return pkl_data


    def _get_embeddings_path(self, dataset_type):
        embeddings_path = os.path.join(Config.EMBEDDINGS_ROOT, 
        Config.EMBEDDINGS_DATASET_NAME, 
        Config.EMBEDDINGS_CROP,
        dataset_type + Config.EMBEDDINGS_FILENAME_TEMPLATE)
        print('Path fetched:', embeddings_path)
        return embeddings_path

    #Adapted from https://github.com/RuohanW/Tasml
    def _index_data(self, pkl_data):
        """Builds an index of images embeddings by class."""
        self._class_image_file_dict = collections.OrderedDict()
        self._image_file_embeddings_dict = collections.OrderedDict()
        for counter, key in enumerate(pkl_data["keys"]): #counter, image file name (e.g. 1072646529445394375-n02099601-n02099601_2439.JPEG)
            _, class_label, image_file = key.split("-")
            self._image_file_embeddings_dict[image_file] = pkl_data["embeddings"][counter]
            if class_label not in self._class_image_file_dict:
                self._class_image_file_dict[class_label] = np.empty(0)
            self._class_image_file_dict[class_label] = np.append(self._class_image_file_dict[class_label], image_file)

        self._validate_index(pkl_data)

        return

    def get_embeddings_for_image_file(self, image_file):
        return self._image_file_embeddings_dict[image_file]

    #Adapted from https://github.com/RuohanW/Tasml
    def _validate_index(self, pkl_data):
        """Performs checks of the data index and image counts per class."""
        n = pkl_data["keys"].shape[0]
        error_message = "{} != {}".format(len(self._image_file_embeddings_dict), n)
        assert len(self._image_file_embeddings_dict) == n, error_message
        error_message = "{} != {}".format(pkl_data["embeddings"].shape[0], n)
        assert pkl_data["embeddings"].shape[0] == n, error_message

        all_class_folders = list(self._class_image_file_dict.keys())
        error_message = "no duplicate class names"
        assert len(set(all_class_folders)) == len(all_class_folders), error_message
        image_counts = set([len(class_images)
                            for class_images in self._class_image_file_dict.values()])
        error_message = ("len(image_counts) should have at least one element but "
                         "is: {}").format(image_counts)
        assert len(image_counts) >= 1, error_message
        assert min(image_counts) > 0    

    #Adapted from https://github.com/RuohanW/Tasml
    def create_db(self, size, num_classes, tr_size, val_size):
        def _build_one_instance_py():
            """Builds a random problem instance using data from specified classes."""
            class_list = list(self._class_image_file_dict.keys())
            sample_count = (tr_size + val_size) #1 + 15
            shuffled_folders = class_list[:]
            random.shuffle(shuffled_folders)
            shuffled_folders = shuffled_folders[:num_classes] #Selecting 5 classes from 97 classes randomly (tieredImageNet 5 way)
            error_message = "len(shuffled_folders) {} is not num_classes: {}".format(
                len(shuffled_folders), num_classes)
            assert len(shuffled_folders) == num_classes, error_message
            image_paths = []
            class_ids = []
            for class_id, class_name in enumerate(shuffled_folders): #Iterate the 5 randomly chosen classes
                all_images = self._class_image_file_dict[class_name]
                all_images = np.random.choice(all_images, sample_count, replace=False) #from all the image embedding file names in one class, select 16 random examples
                error_message = "{} == {} failed".format(len(all_images), sample_count)
                assert len(all_images) == sample_count, error_message
                image_paths.append(all_images) #Contains filenames of the 16 examples
                class_ids.append([[class_id]] * sample_count) #Appends class_id 16 times. This id is given to each randomly chosen class

            label_array = np.array(class_ids, dtype=np.int32)
            path_array = np.array(image_paths)
            task_sig = self._task_signature(path_array, sample_count)

            
            if self._verbose:
                print("task_sig", task_sig.shape)
                print("label_array", label_array.shape)
                print("path_array", path_array.shape)
                
            return task_sig, label_array, path_array

        ret = []
        for i in range(size): #repeat 30,000 times
            ret.append(_build_one_instance_py()) 

        self.db = ret
    
    #Adapted from https://github.com/RuohanW/Tasml
    ''' Finds the kernel mean embedding of dataset '''
    def _task_signature(self, path_array, sig_size):
        embeddings = self._image_file_embeddings_dict
        embedding_array = np.array([[embeddings[image_path] #store the embeddings of the 16 filenames 
                                     for image_path in class_paths[:sig_size]] 
                                    for class_paths in path_array]).astype(np.float64) #Iterate 16 filenames of all 5 classes

        if(self._verbose):
            print("embedding_array: ", embedding_array.shape)
        embedding_array = self.divide_fro_norm(embedding_array) #normalize by dividing by frobenius norm of p=640 components

        mean = np.mean(embedding_array, axis=(0, 1)) #1. class-wise mean (16x640)  --> 2. examples mean (640)

        return mean
    
    def divide_fro_norm(self, a):
        return a / np.linalg.norm(a, axis=-1, keepdims=True) #Norm dimensions (5, 16, 1) 
    
    def save_db(self, path, force=False):
        if path is not None and hasattr(self, "db"):
            if osp.exists(path) and not force:
                print("db exists. Skipping")
            else:
                with open(path, "wb") as f:
                    pickle.dump(self.db, f)