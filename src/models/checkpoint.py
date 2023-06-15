import torch


class ModelCheckpoint:
    def __init__(self, model, file_path):
        self.file_path = file_path
        self.best_loss = float('inf')
        self.model = model

    def __call__(self, epoch, val_loss):
        if val_loss < self.best_loss:
            print('Model saved!')
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), self.file_path)