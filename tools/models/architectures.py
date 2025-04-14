import torch.nn as nn
from inSituML.loss_functions import EarthMoversLoss
from inSituML.ks_models import INNModel


class ModelFinal(nn.Module):
    def __init__(
        self,
        base_network,
        inner_model,
        loss_function_IM=None,
        weight_AE=1.0,
        weight_IM=1.0,
    ):
        super().__init__()

        self.base_network = base_network
        self.inner_model = inner_model
        self.loss_function_IM = loss_function_IM
        self.weight_AE = weight_AE
        self.weight_IM = weight_IM

    def forward(self, x, y):

        loss_AE, loss_ae_reconst, kl_loss, _, encoded = self.base_network(
            x
        )

        # Check if the inner model is an instance of INNModel
        if isinstance(self.inner_model, INNModel):
            # Use the compute_losses function of INNModel
            (loss_IM, l_fit, l_latent, l_rev) = (
                self.inner_model.compute_losses(encoded, y)
            )
            total_loss = (
                loss_AE * self.weight_AE + loss_IM * self.weight_IM
            )

            losses = {
                "total_loss": total_loss,
                "loss_AE": loss_AE * self.weight_AE,
                "loss_IM": loss_IM * self.weight_IM,
                "loss_ae_reconst": loss_ae_reconst,
                "kl_loss": kl_loss,
                "l_fit": l_fit,
                "l_latent": l_latent,
                "l_rev": l_rev,
            }

            return losses
        else:
            # For other types of models, such as MAF
            loss_IM = self.inner_model(inputs=encoded, context=y)
            total_loss = (
                loss_AE * self.weight_AE + loss_IM * self.weight_IM
            )

            losses = {
                "total_loss": total_loss,
                "loss_AE": loss_AE * self.weight_AE,
                "loss_IM": loss_IM * self.weight_IM,
                "loss_ae_reconst": loss_ae_reconst,
                "kl_loss": kl_loss,
            }

            return losses

    def reconstruct(self, x, y, num_samples=1):

        if isinstance(self.inner_model, INNModel):
            lat_z_pred = self.inner_model(x, y, rev=True)
            y = self.base_network.decoder(lat_z_pred)
        else:
            lat_z_pred = self.inner_model.sample_pointcloud(
                num_samples=num_samples, cond=y
            )
            y = self.base_network.decoder(lat_z_pred)

        return y, lat_z_pred