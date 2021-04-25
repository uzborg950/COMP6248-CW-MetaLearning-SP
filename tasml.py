import torch
import torchvision
#
from collections import OrderedDict
#
import TaskClass as TaskClass
import Task as Task

def tasml_epoch_nn_classifier_learn(
        test_net: torch.nn.Module,
        tasks: list[Task.Task],
        alpha_weights: torch.Tensor,
        inner_epochs: int = 1, # Feature not implemented.
        inner_lr: float = 0.001,
        outer_lr: float = 0.001,
        loss_function = torch.nn.CrossEntropyLoss()):
    #
    # Init.
    inner_optimiser = lambda x, g: x - inner_lr * g
    outer_optimiser = torch.optim.Adam(test_net.parameters(), lr=outer_lr)
    #
    def inner_detached_step_on(task: Task.Task) -> OrderedDict: # NOTE: this only does one epoch of SGD per support dataset: more epochs?
        supp_train_inner_loss = loss_function(test_net(task.supp_train), task.supp_targets)
        #
        test_net.zero_grad()
        grads = torch.autograd.grad(supp_train_inner_loss, test_net.parameters(), create_graph=True)
        #
        named_params = OrderedDict()
        for (pname, param), grad in zip(test_net.named_parameters(), grads):
            named_params[pname] = inner_optimiser(param, grad)
        #
        return named_params
    #
    def outer_detached_step_on(task: Task.Task, named_params: OrderedDict):
        backup_named_parameters = test_net.named_parameters
        #
        test_net.named_parameters = named_params
        query_tr_inner_loss = loss_function(test_net(task.query_train), task.query_targets)
        #
        test_net.named_parameters = backup_named_parameters
        #
        return query_tr_inner_loss
    #
    test_net.train()
    outer_optimiser.zero_grad()
    test_net.zero_grad()
    #
    # TASML.
    outer_loss_sum = 0
    for i in range(len(tasks)):
        task = tasks[i]
        query_tr_outer_loss = outer_detached_step_on(task=task, named_params=inner_detached_step_on(task=task))
        outer_loss_sum += alpha_weights[i] * query_tr_outer_loss
    #
    # Extra task.

    #
    outer_loss_sum.backward()
    outer_optimiser.step()

