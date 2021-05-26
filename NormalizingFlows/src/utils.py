import h5py
import torch
import os

def invert_permutation(permutation):
    inv_perm = [0] * len(permutation)

    for i in  range(len(permutation)):
        inv_perm[permutation[i]] = i
    return inv_perm

def permute_data(data, permutation):
    return data[:, permutation.permute()]

def inv_permute_data(data, permutation):
    return data[:, permutation.inv_permute()]

def update_device(device, model, *args):
    model.to(device)
    model.update_device(device)
    for arg in args:
        arg.update_device(device)

def write_to_file(datapath, train_data, validation_data, test_data):
    new_file = h5py.File(datapath, 'w')
    new_file.create_dataset('train', data=train_data.detach().numpy())
    new_file.create_dataset('validation',data=validation_data.detach().numpy())
    new_file.create_dataset('test', data=test_data.detach().numpy())
    new_file.close()

def save_best_model(model, filename=None):
    if not os.path.isdir('trained_models'):
        os.mkdir('trained_models')

    if filename is None:
        filename = 'trained_models/' + str(model) + '_best_model.pth.tar'
    torch.save(model.state_dict(), filename)

def save_checkpoint_model(model, optimizer, losses, filename=None):
    if not os.path.isdir('trained_models'):
        os.mkdir('trained_models')

    if filename is None:
        filename = 'trained_models/' + str(model) + '_checkpoint_model.pth.tar'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
        }, filename)

def load_best_model(model, filename=None):
    if filename is None:
        filename = 'trained_models/' + str(model) + '_best_model.pth.tar'

    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def load_checkpoint_model(model, optimizer, filename=None):
    if filename is None:
        filename = 'trained_models/' + str(model) + '_checkpoint_model.pth.tar'

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint['losses']

    return model, optimizer, losses
