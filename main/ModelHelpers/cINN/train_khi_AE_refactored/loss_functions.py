import torch
import torch.nn as nn
from geomloss import SamplesLoss


class ChamfersLoss(nn.Module):
    """
    Custom loss class for Chamfers Distance.
    Args:
        reduction(str): How to reduce loss from each batch element.
        p(float): value for the p - norm distance to calculate between 
        each vector pair. See also torch.cdist.
    """
    def __init__(reduction='mean',
                 p=2):
        
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
        relative_distances = torch.cdist(x, y, p=p_norm)
        return self.chamfers_distance(relative_distances)

    def chamfers_distance(self, relative_distances):
        """
        Calcuates the chamfers from relative distances.

        Args:
        relative_distances: A tensor containing relative distances between
        the particles.
        """
        loss_per_batch = torch.sum(torch.min(d, -1).values +
                                   torch.min(d, -2).values)

        reduced_loss = getattr(loss_per_batch, self.reduction)()

        return reduced_loss


class EarthMoversLoss(SamplesLoss):
    
    """
    Simple class for earthmovers loss and compatibility in training code of autoencoders.
    Reference: https: // www.kernel - operations.io / geomloss / _auto_examples / comparisons / plot_gradient_flows_2D.html
    """
    def __init__(self, p=1, blur=.01, reduction = "mean"):
        
        super().__init__(loss="sinkhorn", p=p, blur=blur)
        self.reduction = reduction
    
    def forward(self, x, y):
        
        loss_per_batch = super().forward(x, y)
        
        return getattr(loss_per_batch, self.reduction)()
    
