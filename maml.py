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
        backup_named_parameters = test_net.named_parameters
        #
        outer_optimiser.zero_grad()
        test_net.zero_grad()
        #
        outer_loss_sum = 0
        #
        for task in tasks:
            #
            supp_train_inner_loss = loss_function(test_net(task.supp_train), task.supp_targets)
            #
            test_net.zero_grad()
            grads = torch.autograd.grad(supp_train_inner_loss, test_net.parameters(), create_graph=True)
            #
            named_params = OrderedDict()
            for (pname, param), grad in zip(test_net.named_parameters(), grads):
                named_params[pname] = inner_optimiser(param, grad)
            #
            test_net.named_parameters = named_params
            query_tr_inner_loss = loss_function(test_net(task.query_train), task.query_targets)
            test_net.named_parameters = backup_named_parameters
            #
            outer_loss_sum += query_tr_inner_loss
        #
        outer_loss_sum.backward()
        outer_optimiser.step()
