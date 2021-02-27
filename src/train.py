import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import IPython.display as display

def train_forward(
        model, 
        base_dist, 
        train_data, 
        optimizer, 
        epochs=1000, 
        batch_size=16, 
        print_n=100,
        scheduler=None,
    ):

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)  
    train_data.to(device)
    model.to(device)

    batches = DataLoader(dataset=train_data, batch_size=batch_size, 
                            shuffle=True) 

    num_train_data, _ = train_data.size()

    model.train()
    losses = []

    for epoch in range(epochs):
        batch_loss = 0
        for batch in batches:
            z, log_prob = model(batch)
            loss = -torch.mean(log_prob)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            batch_loss += -torch.sum(log_prob)
        losses.append(batch_loss/num_train_data) 

        if epoch % print_n == 0:
            display.clear_output(wait=True)
            print(f"{losses[epoch].item():12.5f}")

    display.clear_output(wait=True)
    print(f"{losses[-1].item():12.5f}")

    model.eval()
    if torch.cuda.is_available():
        dev = "cpu"
        device = torch.device(dev)  
        train_data.to(device)
        model.to(device)

    return losses


def train_backward(
        model, 
        base_dist, 
        target_dist, 
        dim, 
        optimizer, 
        scheduler, 
        epochs=1000, 
        batch_size=16, 
        batches=20, 
        print_n=100,
    ):

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)  
    train_data.to(device)
    model.to(device)

    num_train_data, _ = train_data.size()

    model.train()
    losses = []

    for epoch in range(epochs):
        batch_loss = 0
        for batch in range(batches):
            sample = base_dist.sample((batch_size, dim)) 
            x, log_prob = model(batch)
            target_prob = target_dist.log_prob(x[-1])
            loss = torch.mean(log_prob - target_prob)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss += torch.sum(log_prob-target_prob)

        losses.append(batch_loss/num_train_data) 

        if epoch % print_n == 0:
            display.clear_output(wait=True)
            print(f"{losses[epoch].item():12.5f}")

    display.clear_output(wait=True)
    print(f"{losses[epoch].item():12.5f}")

    model.eval()
    if torch.cuda.is_available():
        dev = "cpu"
        device = torch.device(dev)  
        train_data.to(device)
        model.to(device)

    return losses
