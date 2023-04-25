import torch
import torch.nn as nn

class BerHuLoss(nn.Module):
    """
        Implementation based on the reversed Huber loss function (BerHu) as described in the paper
        "Edge loss functions for deep-learning depth-map" (Paul Sandip et. al, 2022).
    """
    def __init__(self, a: int=0.2):
        super().__init__()
        self.a = torch.tensor(a)
        self.l1loss = nn.L1Loss(reduction="none")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        self.a.to(pred.device)
        absdiff = self.l1loss(pred, gt).view(-1)
        th = self.a * torch.max(absdiff)
        s1 = torch.mean(absdiff[absdiff <= th])
        s2 = torch.mean((absdiff[absdiff > th]**2 + th**2) / (2 * th))
        return (s1 + s2) / 2
