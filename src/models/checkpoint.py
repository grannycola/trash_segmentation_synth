import torch


class ModelCheckpoint:
    def __init__(self, model, file_path, last_best_loss=None):
        self.file_path = file_path
        if not last_best_loss:
            self.best_loss = float('inf')
        else:
            self.best_loss = last_best_loss
        self.model = model

    def __call__(self, epoch, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), self.file_path)
