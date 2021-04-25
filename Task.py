
from __future__ import annotations
try:   
    import torch
    import math
    import torchvision
    import PIL
    import glob
    glob.glob('.')
    import os
    import sys
    import datasetconf as DC
    importlib.reload(TaskClass)
except NameError: # It hasn't been imported yet
    import TaskClass as TaskClass
class Task():

    def __init__(self, task_friendly_name: str, batch_size: int):
        self.task_friendly_name = task_friendly_name
        self.batch_size = batch_size
        self.task_classes: List[TaskClass] = []
        #
        self.supp_train: torch.Tensor
        self.supp_targets: torch.Tensor
        self.query_train: torch.Tensor
        self.query_targets: torch.Tensor

    def add_task_class(self, task_class: TaskClass):
        self.task_classes.append(task_class)


    def reset_test_session(self):
        shape_ts = [i for i in self.task_classes[0].test_imgs[0].shape] # All images of equal size + no empty task classes.
        #
        num_test_samples = sum([len(task_class.test_imgs) for task_class in self.task_classes])
        #
        test_train = torch.empty([num_test_samples] + shape_ts)
        test_targets = torch.empty(num_test_samples, dtype=torch.long)
        #
        i = 0
        for task_class in self.task_classes:
            task_class_id = task_class.class_id
            #
            for class_sample in task_class.test_imgs:
                test_train[i] = class_sample
                test_targets[i] = task_class_id
                i += 1
        #
        # Shuffle.
        new_seq = torch.randperm(num_test_samples)
        self.test_train = test_train[new_seq]
        self.test_targets = test_targets[new_seq]


    def reset_train_session(self):
        shape_ts = [i for i in self.task_classes[0].support_imgs[0].shape] # All images of equal size + no empty task classes.
        #
        num_supp_samples = sum([len(task_class.support_imgs) for task_class in self.task_classes])
        num_query_samples = sum([len(task_class.query_imgs) for task_class in self.task_classes])
        #
        supp_train = torch.empty([num_supp_samples] + shape_ts)
        supp_targets = torch.empty(num_supp_samples, dtype=torch.long)
        #
        query_train = torch.empty([num_query_samples] + shape_ts)
        query_targets = torch.empty(num_query_samples, dtype=torch.long)
        #
        i = 0
        j = 0
        for task_class in self.task_classes:
            task_class_id = task_class.class_id
            #
            for class_sample in task_class.support_imgs:
                supp_train[i] = class_sample
                supp_targets[i] = task_class_id
                i += 1
            #
            for class_sample in task_class.query_imgs:
                query_train[j] = class_sample
                query_targets[j] = task_class_id
                j += 1
        #
        #Shuffle
        if len(self.task_classes) > 1:
            new_seq = torch.randperm(num_supp_samples)
            self.supp_train = supp_train[new_seq]
            self.supp_targets = supp_targets[new_seq]
            #
            new_seq = torch.randperm(num_query_samples)
            self.query_train = query_train[new_seq]
            self.query_targets = query_targets[new_seq]
        else:
            self.supp_train = supp_train
            self.supp_targets = supp_targets
            #
            self.query_train = query_train
            self.query_targets = query_targets


LEN_ALL_AVAILABLE = sys.maxsize


def create_task_given(
    task_friendly_name: str,
    dataset_name: str,
    class_names: list[str],
    len_support_dataset: int,
    len_query_dataset: int,
    transformer: torchvision.transforms.transforms.Compose,
    img_size: int,
    let_test_dataset: int = sys.maxsize,
    batch_size: int = 4,
    start_class_id: int = 0):
    #
    # Create task class container obj.
    task = Task(task_friendly_name=task_friendly_name, batch_size=batch_size)
    #
    for class_id in range(start_class_id, start_class_id + len(class_names), 1):
        class_name = class_names[class_id - start_class_id]
        task.add_task_class(TaskClass.create_task_class_given(
            class_id=class_id,
            class_friendly_name=class_name,
            support_path=DC.train_path_of(dataset_name, class_name=class_name),
            query_path=DC.val_path_of(dataset_name, class_name=class_name),
            test_path=DC.test_path_of(dataset_name, class_name=class_name),
            transformer=transformer,
            img_size=img_size,
            classname=class_name,
            len_support_dataset=len_support_dataset,
            len_query_dataset=len_query_dataset,
            let_test_dataset=let_test_dataset))
    #
    return task

