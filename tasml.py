from __future__ import annotations
#
import sys
#
import torch
import torchvision
#
from collections import OrderedDict
#
import TaskClass as TaskClass
import Task as Task

def tasml_nn_classifier_learn(
        test_net: torch.nn.Module,
        tasks: list[Task.Task],
        target_task: Task.Task,
        alpha_weights: torch.Tensor,
        convergence_diff: float = 0.0001,
        max_meta_epochs = 10,
        inner_epochs: int = 1,
        inner_lr: float = 0.001,
        outer_lr: float = 0.001,
        loss_function = torch.nn.CrossEntropyLoss()):
    #
    # Init.
    inner_optimiser = lambda x, g: x - inner_lr * g
    outer_optimiser = torch.optim.Adam(test_net.parameters(), lr=outer_lr)
    #
    # Common initialization state cache.
    backup_named_parameters = OrderedDict()
    #
    def cache_theta():
        for name, params in test_net.named_parameters():
            backup_named_parameters[name] = params.clone()
    #
    def load_named_params_of(named_params: OrderedDict):
        for name, params in test_net.named_parameters():
            params.data.copy_(named_params[name])
    #
    def load_theta():
        load_named_params_of(backup_named_parameters)
    #
    def load_theta_prime(theta_prime: OrderedDict):
        load_named_params_of(theta_prime)
    #
    test_net.train()
    #
    # Meta training status.
    meta_epochs_counter = 0
    prev_loss = 0 # Accumulated loss 2 rounds ago.
    last_loss = sys.maxsize # Accumulated loss last round.
    # TASML.
    losses = []
    while (abs(prev_loss - last_loss) > convergence_diff and meta_epochs_counter < max_meta_epochs):
        #
        # Store this epoch's theta.
        cache_theta()
        #
        outer_optimiser.zero_grad()
        test_net.zero_grad()
        #
        outer_loss_sum = 0
        #
        for task_index in range(len(tasks)):
            task = tasks[task_index]
            #
            # Inner loss.
            for _ in range(inner_epochs):
                supp_train_inner_loss = loss_function(test_net(task.supp_train), task.supp_targets)
                test_net.zero_grad()
                grads = torch.autograd.grad(supp_train_inner_loss, test_net.parameters(), create_graph=True)
                #
                # Compute theta prime.
                named_params = OrderedDict()
                for (pname, param), grad in zip(test_net.named_parameters(), grads):
                    named_params[pname] = inner_optimiser(param, grad)
                # Load theta prime.
                load_theta_prime(named_params)
            #
            # Other loss
            outer_loss_sum += alpha_weights[task_index] * loss_function(test_net(task.query_train), task.query_targets)
            #
            # Reload theta.
            load_theta()
        #
        #
        # Extra task L((Alg(D), D) | D = data from tasked task, only from support dataset.
        for _ in range(inner_epochs):
            supp_train_inner_loss = loss_function(test_net(target_task.supp_train), target_task.supp_targets)
            test_net.zero_grad()
            grads = torch.autograd.grad(supp_train_inner_loss, test_net.parameters(), create_graph=True)
            #
            # Compute theta prime.
            named_params = OrderedDict()
            for (pname, param), grad in zip(test_net.named_parameters(), grads):
                named_params[pname] = inner_optimiser(param, grad)
            # Load theta prime.
            load_theta_prime(named_params)
        #
        outer_loss_sum += loss_function(test_net(target_task.supp_train), target_task.supp_targets)
        #
        # Reload theta.
        load_theta()
        #
        print("Epoch " + str(meta_epochs_counter) + ": "+ str(outer_loss_sum.item()))
        losses.append(outer_loss_sum.item())
        outer_loss_sum.backward()
        outer_optimiser.step()
        #
        prev_loss = last_loss
        last_loss = outer_loss_sum.item()
        meta_epochs_counter += 1
    return losses
