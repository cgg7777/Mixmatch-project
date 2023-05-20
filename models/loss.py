import torch
import torch.nn.functional as F

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u = 75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u
