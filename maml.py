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

def maml_nn_classifier_learn(
        test_net: torch.nn.Module,
        tasks: list[Task.Task],
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
    test_net.train()
    #
    # Meta training status.
    meta_epochs_counter = 0
    prev_loss = 0 # Accumulated loss 2 rounds ago.
    last_loss = sys.maxsize # Accumulated loss last round.
    # MAML.
    while (abs(prev_loss - last_loss) > convergence_diff and meta_epochs_counter < max_meta_epochs):
        #
        # Store this epoch's theta.
        backup_named_parameters = OrderedDict()
        for name, params in test_net.named_parameters():
            backup_named_parameters[name] = params.clone()
        #
        outer_optimiser.zero_grad()
        test_net.zero_grad()
        #
        outer_loss_sum = 0
        #
        for task in tasks:
            #
            # Inner loss.
            supp_train_inner_loss = loss_function(test_net(task.supp_train), task.supp_targets)
            test_net.zero_grad()
            grads = torch.autograd.grad(supp_train_inner_loss, test_net.parameters(), create_graph=True)
            #
            # Compute theta prime.
            named_params = OrderedDict()
            for (pname, param), grad in zip(test_net.named_parameters(), grads):
                named_params[pname] = inner_optimiser(param, grad)
            # Load theta prime.
            for name, params in test_net.named_parameters():
                params.data.copy_(named_params[name])
            #
            # Other loss
            outer_loss_sum += loss_function(test_net(task.query_train), task.query_targets)
            #
            # Reload theta.
            for name, params in test_net.named_parameters():
                params.data.copy_(backup_named_parameters[name])
        #
        outer_loss_sum.backward()
        outer_optimiser.step()
        #
        prev_loss = last_loss
        last_loss = outer_loss_sum.item()
        meta_epochs_counter += 1
