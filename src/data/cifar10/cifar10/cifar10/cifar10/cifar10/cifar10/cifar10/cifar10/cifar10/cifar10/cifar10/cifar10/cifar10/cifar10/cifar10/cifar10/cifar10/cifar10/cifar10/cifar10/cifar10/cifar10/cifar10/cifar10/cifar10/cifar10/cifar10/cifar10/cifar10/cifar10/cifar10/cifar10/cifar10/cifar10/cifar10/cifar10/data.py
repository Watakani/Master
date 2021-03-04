import torch
import torch.nn

class Dataset:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_data = None
        self.test_data = None

    def get_training_data(self):
        return self.train_data.to(self.device)

    def get_test_data(self):
        return self.test_data.to(self.device)

    def update_device(self, device):
        self.device = device

