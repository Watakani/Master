import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import ray
from ray import tune

from .utils import save_best_model, save_checkpoint_model

import IPython.display as display

def train_forward(
        model, 
        train_data, 
        optimizer, 
        epochs=1000, 
        batch_size=16, 
        print_n=10,
        save_checkpoint=False,
        save_best=True,
        burn_in=100
    ):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_data = train_data.to(device)
    

    batches = DataLoader(dataset=train_data, batch_size=batch_size, 
                            shuffle=True) 

    num_train_data, _ = train_data.size()
    name = str(model)

    model.train()
    losses = []
    best_loss = None

    for epoch in range(epochs):
        batch_loss = 0
        for index, batch in enumerate(batches):
            model.zero_grad()
            z, log_prob = model(batch)
            loss = -torch.mean(log_prob)

            loss.backward()
            optimizer.step()

            batch_loss += -torch.sum(log_prob)
            losses.append(loss.item())

            if (index * (epoch+1)) % print_n == 0:
                display.clear_output(wait=True)
                print(name, "Epoch: {}".format(epoch), "Batch number: {}".format(index), 
                        f"{loss.item():12.5f}")

        epoch_loss = batch_loss/num_train_data
        if save_best and epoch > burn_in:
            if best_loss is None or epoch_loss < best_loss:
                save_best_model(model)

    display.clear_output(wait=True)
    print("Finished training. Loss for last epoch " + name + ':', f"{epoch_loss.item():12.5f}")

    model.eval()
    if save_checkpoint:
        save_checkpoint_model(model, optimizer, losses)

    return losses

def train_forward_with_tuning(
        config,
        model=None, 
        dataset=None, 
        epochs=1000, 
        batch_size=16, 
        print_n=100,
        optimizer=None,
        scheduler=None,
        validation=True
    ):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    train_data = dataset.get_training_data()
    if validation:
        validation_data = dataset.get_validation_data()
        num_val_batch = dataset.valid_n

    batches = DataLoader(dataset=train_data, batch_size=batch_size, 
                            shuffle=True) 

    num_train_data, _ = train_data.size()
    name = str(model)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    model.train()
    losses = []

    for epoch in range(epochs):
        batch_loss = 0
        for batch in batches:
            model.zero_grad()
            z, log_prob = model(batch)
            loss = -torch.mean(log_prob)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if validation:
                z, log_prob = model(validation_data)
                val_loss = -torch.mean(log_prob)
                tune.report(loss=(losses[epoch].detach().cpu().numpy()))
            else:
                tune.report(loss=(losses[epoch].detach().cpu().numpy()))

    model.eval()

def train_backward(
        model, 
        base_dist, 
        target_dist, 
        optimizer, 
        epochs=1000, 
        batch_size=16, 
        batches=20, 
        print_n=100,
        save_checkpoint=False,
        save_best=True,
        burn_in=100
    ):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    name = str(model)

    model.train()
    losses = []
    best_loss = None

    for epoch in range(epochs):
        batch_loss = 0
        for batch in range(batches):
            model.zero_grad()

            sample = base_dist.sample(batch_size)
            x, log_prob = model(batch)
            target_prob = target_dist.log_prob(x[-1])
            loss = torch.mean(log_prob - target_prob)

            loss.backward()
            optimizer.step()

            batch_loss += torch.sum(log_prob-target_prob)
            losses.append(loss.item())

            if (index * (epoch+1)) % print_n == 0:
                display.clear_output(wait=True)
                print(name, "Epoch: {}".format(epoch), "Batch number: {}".format(index), 
                        f"{loss.item():12.5f}")

        epoch_loss = batch_loss/(batch_size * batches)
        if save_best and epoch > burn_in:
            if best_loss is None or epoch_loss < best_loss:
                save_best_model(model)

    display.clear_output(wait=True)
    print("Finished training. Loss for last epoch " + name + ':', f"{epoch_loss.item():12.5f}")

    model.eval()

    if save_checkpoint:
        save_checkpoint_model(model, optimizer, losses)

    return losses

def train_backward_with_tuning(
        config,
        model=None, 
        base_dist=None,
        target_dist=None,
        epochs=1000, 
        batch_size=16, 
        print_n=100,
        name=None,
        optimizer=None
    ):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    model.train()
    losses = []

    for epoch in range(epochs):
        batch_loss = 0
        for batch in range(batches):
            model.zero_grad()

            sample = base_dist.sample(batch_size)
            x, log_prob = model(batch)
            target_prob = target_dist.log_prob(x[-1])
            loss = torch.mean(log_prob - target_prob)

            batch_loss += torch.sum(log_prob - target_prob)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            sample = base_dist.sample(batch_size)
            x, log_prob = model(validation_data)
            target_prob = target_dist.log_prob(x[-1])
            val_loss = torch.mean(log_prob - target_prob)
            tune.report(loss=(val_loss.detach().cpu().numpy()))

    model.eval()
