import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM


class BasicGraspModule(nn.Module):
    """
    An abstract module for grasp quality prediction.
    """

    def __init__(self):
        super(BasicGraspModule, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, x_in, y_desired):
        qm_desired, drx_desired, dry_desired, drz_desired = y_desired
        qm_predict, drx_predict, dry_predict, drz_predict = self(x_in)

        qm_loss = F.mse_loss(qm_predict, qm_desired)
        drx_loss = F.mse_loss(drx_predict, drx_desired)
        dry_loss = F.mse_loss(dry_predict, dry_desired)
        drz_loss = F.mse_loss(drz_predict, drz_desired)

        return {
            'loss': qm_loss + drx_loss + dry_loss + drz_loss,
            'losses': {
                'qm_loss': qm_loss,
                'drx_loss': drx_loss,
                'dry_loss': dry_loss,
                'drz_loss': drz_loss
            },
            'pred': {
                'qm_predict': qm_predict,
                'drx_predict': drx_predict,
                'dry_predict': dry_predict,
                'drz_predict': drz_predict
            }
        }

    def predict(self, x_in):
        qm_predict, drx_predict, dry_predict, drz_predict = self(x_in)
        return {
            'qm': qm_predict,
            'drx': drx_predict,
            'dry': dry_predict,
            'drz': drz_predict
        }


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.cbam = CBAM(out_channels, 16)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.bn2(self.conv2(x))

        x = F.leaky_relu(x)
        x = self.cbam(x)

        return x + x_in
