import os
import numpy as np
import torch
import torchvision
import copy
#
import config.config_flags as Config
import data_load.data_provider as dp
import runner as runner
import utils.task_helper as th
import utils.helper as helper
import TaskClass as TaskClass
import Task as Task
import TestNets as TestNets
import maml as MAML
import tasml as TASML
import testing_routines as TESTING_ROUTINES


#Set tensor device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#######################################################################################################################################
# Generate Task DB or load from filesystem

#print("Generating tasks and compute their alpha weights")
#runner.populate_db()

#Perform top-m filtering
print("Filtering top-m alpha weights and training tasks")
alpha_weights, train_db = runner.top_m_filtering()


#Get target task from filesystem
test_path = helper.get_task_dataset_path("test")
test_db = runner.unpickle(test_path)


#Get train data embeddings
print("Fetching train embeddings")
train_provider = dp.DataProvider("train", debug=False, verbose=False)
train_tr_size = Config.TRAINING_NUM_OF_EXAMPLES_PER_CLASS
train_val_size = Config.VALIDATION_NUM_OF_EXAMPLES_PER_CLASS
print("Generating training tasks")
num_test_tasks = alpha_weights.shape[1]
train_tasks = []
for n in range(num_test_tasks):
    print("Generating top m training tasks for test task " + str(n))
    train_tasks.append(th.generate_tasks(train_db[n], train_provider, train_tr_size, train_val_size, device))
del train_db, train_provider #Free up space


#Get train data embeddings
print("Fetching test embeddings")
test_provider = dp.DataProvider("test", debug=False, verbose=False)
test_tr_size = Config.TRAINING_NUM_OF_EXAMPLES_PER_CLASS
test_val_size = Config.TEST_VALIDATION_NUM_OF_EXAMPLES_PER_CLASS
print("Generating test tasks")
test_tasks = th.generate_tasks(test_db, test_provider, test_tr_size, test_val_size, device) # Target tasks with only tests populated
del test_db, test_provider #Free up space


# Create network test instance.
def get_test_net():
    return TestNets.MAMLModule1(input_len=640, n_classes=Config.NUM_OF_CLASSES)


# Iterate each test task
#Fetch the training tasks and weights for the test task
for test_task_num, target_task in enumerate(test_tasks):
    alpha_weights_for_target = alpha_weights[:,test_task_num]
    training_tasks_for_target = train_tasks[test_task_num] #returns list

    # Remaps to utilization of only training images for support and query.
    training_target_task = Task.Task(task_friendly_name=target_task.task_friendly_name, batch_size=target_task.batch_size)
    training_target_task.supp_train = target_task.supp_train
    training_target_task.supp_targets = target_task.supp_targets
    training_target_task.query_train = target_task.supp_train
    training_target_task.query_targets = target_task.supp_targets
    #
    test_target_task = target_task

    # Test nn modules (shared initial state).
    test_net_base = get_test_net()
    #
    test_net_maml = get_test_net()
    test_net_maml.load_state_dict(copy.deepcopy(test_net_base.state_dict()))
    #
    test_net_tasml = get_test_net()
    test_net_tasml.load_state_dict(copy.deepcopy(test_net_base.state_dict()))

    TESTING_ROUTINES.run_baselearner(test_net_base, training_tasks_for_target, training_target_task, test_target_task)
    TESTING_ROUTINES.run_maml(test_net_maml, training_tasks_for_target, training_target_task, test_target_task) # TODO: no top m filtering for MAML
    TESTING_ROUTINES.run_tasml(test_net_tasml, training_tasks_for_target, training_target_task, alpha_weights_for_target, test_target_task)
