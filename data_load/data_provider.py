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
    importlib.reload(Config)
except NameError: # It hasn't been imported yet
        import config.config_flags as Config

#Dataset type: train, val, test
class DataProvider(object):
    def __init__(self, dataset_type, debug=False):
        self._dataset_type = dataset_type
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