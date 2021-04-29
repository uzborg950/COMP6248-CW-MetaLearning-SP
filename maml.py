from __future__ import annotations
import torch
import torchvision
#
from collections import OrderedDict
#
import TaskClass as TaskClass
import Task as Task

# NOTE: this only does one epoch of SGD per support dataset: more epochs?
def maml_nn_classifier_learn(
        test_net: torch.nn.Module,
        tasks: list[Task.Task],
        inner_epochs: int = 1,
        meta_epochs: int = 1,
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
    # MAML.
    for _ in range(meta_epochs): # TODO: change to 'convergence'?.
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
            # Reload theta.
            for name, params in test_net.named_parameters():
                params.data.copy_(backup_named_parameters[name])
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
        outer_loss_sum.backward()
        outer_optimiser.step()

