import torch
from torchmetrics import JaccardIndex


def IoU(preds, masks, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jaccard_score = JaccardIndex(task='multiclass',
                                 num_classes=num_classes,
                                 ignore_index=0).to(device)
    return jaccard_score(preds, torch.squeeze(masks, dim=1))
