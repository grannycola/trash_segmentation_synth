import torch


class ModelCheckpoint:
    def __init__(self, model, file_path, last_best_loss=None, last_best_metric=None):
        self.file_path = file_path
        if not last_best_loss:
            self.best_loss = float('inf')
        else:
            self.best_loss = last_best_loss
            self.best_metric = last_best_metric
            print('Last best loss: ' + str(float(self.best_loss)))
            print('Last best metric: ' + str(float(self.best_metric)))
        self.model = model

    def __call__(self, epoch, loss, metric):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_metric = metric
            state = {
                'model': self.model.state_dict(),
                'best_loss': self.best_loss,
                'best_metric': self.best_metric,
            }
            torch.save(state, self.file_path)
