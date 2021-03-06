from __future__ import annotations
#
import os
import numpy as np
import torch
import torchvision
import PIL
import copy
import time
import sys
import csv
#
import config.config_flags as Config
import data_load.data_provider as dp
import runner as runner
import utils.task_helper as th
import utils.helper as helper
import datasetconf as DC
import TaskClass as TaskClass
import Task as Task
import TestNets as TestNets
import maml as MAML
import tasml as TASML
import baselearner as BASELEARNER

OUTPUT_FILE_NAME = 'Log_Tiered_True_5w5s_500_5_MAMLModule1'

def acc_of_training_module_on(test_net: torch.nn.Module, test_task: Task):
    acc = 0
    with torch.no_grad():
        test_net.eval()
        acc = (torch.sum(torch.argmax(test_net(test_task.query_train), dim=1) == test_task.query_targets) / test_task.query_targets.shape[0]).item() * 100
        test_net.train()
        test_net.zero_grad()
    return round(acc, 2)


def finetune_module_on(test_net: torch.nn.Module, fine_tuning_task, isMetaFinetuned: bool):
    if isMetaFinetuned:
        MAML.maml_nn_classifier_learn(
            test_net = test_net,
            tasks=[fine_tuning_task],
            # Tunables:
            convergence_diff=0.0001,
            max_meta_epochs=500,
            inner_epochs=1
        )
    else:
        BASELEARNER.base_nn_classifier_finetune(
            test_net=test_net,
            fine_tuning_task=fine_tuning_task,
            # Tunables:
            convergence_diff=0.0001,
            max_epochs=500
        )


def run_baselearner(test_net: torch.nn.Module, training_tasks: list[Task], target_task: Task, test_task: Task):
    acc0 = acc_of_training_module_on(test_net, test_task)
    t0 = time.time()
    # Train
    train_losses = BASELEARNER.base_nn_classifier_learn(
        test_net=test_net,
        training_tasks=training_tasks,
        # Tunables:
        convergence_diff=0.0001,
        max_epochs=500
    )
    t1 = time.time()
    tr0 = t1 - t0
    acc1 = acc_of_training_module_on(test_net, test_task)
    t1 = time.time()
    #
    finetune_module_on(test_net, fine_tuning_task=target_task, isMetaFinetuned=False)
    t2 = time.time()
    tr1 = t2 - t1
    trtot = t2 - t0
    acc2 = acc_of_training_module_on(test_net, test_task)
    #
    log = ', '.join([str(x) for x in ['BASE', target_task.task_friendly_name, acc0, '-', acc1, acc2, tr0, tr1, '-', trtot]]) + '\n'
    print(log)
    with open(OUTPUT_FILE_NAME + '_BASE.csv', "a") as file_object:
        file_object.write(log)
    with open(OUTPUT_FILE_NAME +"_"+ target_task.task_friendly_name+'_LOSS_BASE.csv', "a") as file_object:
        write = csv.writer(file_object)
        write.writerow(train_losses)


def run_maml(test_net: torch.nn.Module, training_tasks: list[Task], target_task: Task, test_task: Task, isMetaFinetuned=True):
    acc0 = acc_of_training_module_on(test_net, test_task)
    t0 = time.time()
    # Train.
    train_losses = MAML.maml_nn_classifier_learn(
        test_net = test_net,
        tasks=training_tasks,
        # Tunables:
        convergence_diff=0.0001,
        max_meta_epochs=500,
        inner_epochs=1
    )
    t1 = time.time()
    tr0 = t1 - t0
    acc1 = acc_of_training_module_on(test_net, test_task)
    t1 = time.time()
    # Finetune/Meta.
    finetune_module_on(test_net, target_task, isMetaFinetuned)
    t2 = time.time()
    tr1 = t2 - t1
    trtot = t2 - t0
    acc2 = acc_of_training_module_on(test_net, test_task)
    #
    log = ', '.join([str(x) for x in ['MAML', target_task.task_friendly_name, acc0, '-', acc1, acc2, '-', tr0, tr1, trtot]]) + '\n'
    print(log)
    with open(OUTPUT_FILE_NAME + '_MAML.csv', "a") as file_object:
        file_object.write(log)
    with open(OUTPUT_FILE_NAME +"_"+ target_task.task_friendly_name+'_LOSS_MAML.csv', "a") as file_object:
        write = csv.writer(file_object)
        write.writerow(train_losses)

def run_tasml(test_net: torch.nn.Module, training_tasks: list[Task], target_task: Task, alpha_weights: torch.Tensor, test_task: Task, isMetaFinetuned=True):
    acc0 = acc_of_training_module_on(test_net, test_task)
    t0 = time.time()
    # Warm-start.
    train_losses_warmstart = MAML.maml_nn_classifier_learn(
        test_net = test_net,
        tasks=training_tasks,
        # Tunables:
        convergence_diff=0.0001,
        max_meta_epochs=30,
        inner_epochs=1
    )
    t1 = time.time()
    tr0 = t1 - t0
    acc1 = acc_of_training_module_on(test_net, test_task)
    t1 = time.time()
    # Tasml.
    train_losses_tasml = TASML.tasml_nn_classifier_learn(
        test_net = test_net,
        tasks=training_tasks,
        target_task=target_task,
        alpha_weights=alpha_weights,
        # Tunables:
        convergence_diff=0.0001,
        max_meta_epochs=500,
        inner_epochs=1
    )
    t2 = time.time()
    tr1 = t2 - t1
    acc2 = acc_of_training_module_on(test_net, test_task)
    t2 = time.time()
    # Finetune/Meta.
    finetune_module_on(test_net, target_task, isMetaFinetuned)
    t3 = time.time()
    tr2 = t3 - t2
    trtot = t3 - t0
    #
    acc3 = acc_of_training_module_on(test_net, test_task)
    #
    log = ', '.join([str(x) for x in ['TASML', target_task.task_friendly_name, acc0, acc1, acc2, acc3, tr0, tr1, tr2, trtot]]) + '\n'
    print(log)
    with open(OUTPUT_FILE_NAME + '_TASML.csv', "a") as file_object:
        file_object.write(log)
    with open(OUTPUT_FILE_NAME +"_"+ target_task.task_friendly_name+'_WARMSTART_LOSS_MAML.csv', "a") as file_object:
        write = csv.writer(file_object)
        write.writerow(train_losses_warmstart)
    with open(OUTPUT_FILE_NAME +"_"+ target_task.task_friendly_name+'_LOSS_TASML.csv', "a") as file_object:
        write = csv.writer(file_object)
        write.writerow(train_losses_tasml)

