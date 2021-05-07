from __future__ import annotations
#
import sys
#
import torch
import Task as Task
import TaskClass as TaskClass


def base_nn_classifier_learn(
    test_net: torch.nn.Module,
    training_tasks: list[Task],
    max_epochs=10,
    lr=0.001,
    convergence_diff=0.0001,
    loss_function=torch.nn.CrossEntropyLoss()):
    # Init.
    optimiser = torch.optim.Adam(test_net.parameters(), lr=lr)
    #
    test_net.train()
    test_net.zero_grad()
    #
    # Base training status.
    epochs_counter = 0
    prev_loss = 0 # Loss 2 rounds ago.
    last_loss = sys.maxsize # Loss last round.
    #
    losses = []
    while (abs(prev_loss - last_loss) > convergence_diff and epochs_counter < max_epochs):
        loss_summ = 0
        for task in training_tasks:

            # Train on support dataset.
            optimiser.zero_grad()
            loss = loss_function(test_net(task.supp_train), task.supp_targets)
            loss.backward()
            optimiser.step()

            loss_summ += loss.item()

            # Train on query dataset.
            optimiser.zero_grad()
            loss = loss_function(test_net(task.query_train), task.query_targets)
            loss.backward()
            optimiser.step()

            loss_summ += loss.item()
        #
        print("Epoch " + str(epochs_counter) + ": "+ str(loss_summ))
        losses.append(loss_summ)
        prev_loss = last_loss
        last_loss = loss_summ
        epochs_counter += 1
    return losses

def base_nn_classifier_finetune(
    test_net: torch.nn.Module,
    fine_tuning_task: Task,
    max_epochs=10,
    lr=0.001,
    convergence_diff=0.0001,
    loss_function=torch.nn.CrossEntropyLoss()):
    # Init.
    optimiser = torch.optim.Adam(test_net.parameters(), lr=lr)
    #
    test_net.train()
    test_net.zero_grad()
    #
    # Base training status.
    epochs_counter = 0
    prev_loss = 0 # Accumulated loss 2 rounds ago.
    last_loss = sys.maxsize # Accumulated loss last round.
    #
    while (abs(prev_loss - last_loss) > convergence_diff and epochs_counter < max_epochs):
        # Train on support dataset.
        optimiser.zero_grad()
        loss = loss_function(test_net(fine_tuning_task.supp_train), fine_tuning_task.supp_targets)
        loss.backward()
        optimiser.step()
        #
        prev_loss = last_loss
        last_loss = loss
        epochs_counter += 1
