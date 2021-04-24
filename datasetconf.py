ROOT_FOLDER = 'datasets/'

def train_path_of(dataset_name: str, class_name=''):
    return ROOT_FOLDER + dataset_name + '/data/train/' + class_name + '/'


def val_path_of(dataset_name: str, class_name=''):
    return ROOT_FOLDER + dataset_name + '/data/valid/' + class_name + '/'


def test_path_of(dataset_name: str, class_name=''):
    return ROOT_FOLDER + dataset_name + '/data/test/' + class_name + '/'


def is_img_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
