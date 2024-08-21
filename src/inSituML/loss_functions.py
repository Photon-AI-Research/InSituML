import torch
import torch.nn as nn
from geomloss import SamplesLoss
import traceback

try:
    from .ChamferDistancePytorch.chamfer6D import (
        dist_chamfer_6D as dist_chamfer_6D,
    )
    from .ChamferDistancePytorch.chamfer3D import (
        dist_chamfer_3D as dist_chamfer_3D,
    )

    class ChamfersLossOptimized(nn.Module):
        def __init__(self, property_="momentum_force"):

            super().__init__()

            if property_ == "momentum_force":
                self.chm_obj = dist_chamfer_6D.chamfer_6DDist()
            elif property_ != "all":
                self.chm_obj = dist_chamfer_3D.chamfer_6DDist()
            else:
                raise ValueError(
                    (
                        "Either wrong property_ name or property_ 'all' can't"
                        + " be used with ChamfersLossOptimized"
                    )
                )

        def forward(self, x, y):
            dist1, dist2, idx1, idx2 = self.chm_obj(x, y)
            loss = torch.mean(dist1) + torch.mean(dist2)
            return loss

except Exception:
    traceback.print_exc()


class ChamfersLossDiagonal(nn.Module):
    """
    Custom loss class for Chamfers Distance taken with diagonal
    things to metrics.
    Taken from https://github.com/lingjiekong/CS236Project/blob/
                                    eval_metric/metrics/evaluation_metrics.py
    Args:
        reduction(str): How to reduce loss from each batch element.
        p(float): value for the p - norm distance to calculate between
        each vector pair. See also torch.cdist.
    """

    def __init__(self, reduction="mean", p=2):

        super().__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, x, y):
        """

        Args:
            x(Tensor): Output of the model.
            y(Tensor): Ground truth values
        """
        dl, dr = self.chamfers_distance(x, y)
        cd_loss = dl.mean(dim=1) + dr.mean(dim=1)
        return cd_loss

    def chamfers_distance(self, a, b):
        x, y = a, b
        _, num_points, _ = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).to(a).long()
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P.min(1)[0], P.min(2)[0]


class ChamfersLoss(nn.Module):
    """
    Custom loss class for Chamfers Distance.
    Args:
        reduction(str): How to reduce loss from each batch element.
        p(float): value for the p - norm distance to calculate between
        each vector pair. See also torch.cdist.
    """

    def __init__(self, reduction="mean", p=2):

        super().__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, x, y):
        """
        First calculate the relative distance between all the points and
        then calculates chamfers distance.
        Args:
            x(Tensor): Output of the model.
            y(Tensor): Ground truth values
        """
        relative_distances = torch.cdist(x, y, p=self.p)
        return (
            self.chamfers_distance(relative_distances)
            / relative_distances.shape[-1]
        )

    def chamfers_distance(self, relative_distances):
        """
        Calcuates the chamfers from relative distances.

        Args:
        relative_distances: A tensor containing relative distances between
        the particles.
        """
        a = torch.min(relative_distances, -1).values
        b = torch.min(relative_distances, -2).values
        loss_per_batch = torch.sum(a, -1) + torch.sum(b, -1)

        reduced_loss = getattr(loss_per_batch, self.reduction)()

        return reduced_loss


class EarthMoversLoss(SamplesLoss):
    """
    Simple class for earthmovers loss and compatibility in training code
     of autoencoders.
    Reference: https: // www.kernel - operations.io / geomloss /
                    _auto_examples / comparisons / plot_gradient_flows_2D.html
    """

    def __init__(self, p=1, blur=0.01, backend="auto", reduction="mean"):

        super().__init__(loss="sinkhorn", p=p, blur=blur, backend=backend)
        self.reduction = reduction

    def forward(self, x, y):
        loss_per_batch = super().forward(x.contiguous(), y.contiguous())

        return getattr(loss_per_batch, self.reduction)()
