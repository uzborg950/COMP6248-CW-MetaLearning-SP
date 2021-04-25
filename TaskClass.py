from __future__ import annotations
import torch
import math
import torchvision
import PIL
import glob
glob.glob('.')
import os
import sys
import datasetconf as DC
class TaskClass():

    def __init__(self, class_id: int, class_friendly_name: str = None):
        self.support_imgs: list[torch.Tensor]
        self.query_imgs: list[torch.Tensor]
        self.test_imgs: list[torch.Tensor]
        self.class_id = class_id
        self.class_friendly_name = str(class_id) if class_friendly_name == None else class_friendly_name


    def set_support_dataset(self, support_imgs: List[torch.Tensor]):
        self.support_imgs = support_imgs


    def set_query_dataset(self, query_imgs: List[torch.Tensor]):
        self.query_imgs = query_imgs


    def set_test_dataset(self, test_imgs: List[torch.Tensor]):
        self.test_imgs = test_imgs


# def __get_batched_data_list__(lst: list[torch.Tensor], batch_size: int):
#     blst = []
#     num_ts = len(lst)
#     shape_ts = [i for i in lst[0].shape]
#     i = 0
#     while i < num_ts:
#         if batch_size <= num_ts - i:
#             this_batch = torch.empty([batch_size] + shape_ts)
#             for j in range(batch_size):
#                 this_batch[j] = lst[i]
#                 i += 1
#             blst.append(this_batch)
#         else:
#             break
#     return blst


# def __get_var_size_batched_data_list__(lst: list[torch.Tensor], batch_size: int):
#     blst = []
#     num_ts = len(lst)
#     shape_ts = [i for i in lst[0].shape]
#     i = 0
#     while i < num_ts:
#         this_batch_size = min(batch_size, num_ts - i)
#         this_batch = torch.empty([this_batch_size] + shape_ts)
#         for j in range(this_batch_size):
#             this_batch[j] = lst[i]
#             i += 1
#         blst.append(this_batch)
#     return blst


def create_task_class_given(
        class_id: int,
        class_friendly_name: str,
        support_path: str,
        query_path: str,
        test_path: str,
        transformer: torchvision.transforms.transforms.Compose,
        img_size: int,
        classname: str,
        len_support_dataset: int,
        len_query_dataset: int,
        let_test_dataset: int = sys.maxsize):
    #
    # Loads all n images in path through PIL.Imgae.open and transforms them.
    def load_imgs(path: str, n=sys.maxsize):
        return [transformer(PIL.Image.open(imf)) for imf in
                    [ path + f for f in
                    os.listdir(path)
                    if DC.is_img_file(f)
                    ][:n]]
    #
    # Build TaskClass object.
    task_class = TaskClass(class_id, class_friendly_name=class_friendly_name)
    #
    # Support set.
    task_class.set_support_dataset(load_imgs(support_path, len_support_dataset))
    # Query set.
    task_class.set_query_dataset(load_imgs(query_path, len_query_dataset))
    # Test set (loads all of them).
    task_class.set_test_dataset(load_imgs(test_path, let_test_dataset))
    #
    return task_class
