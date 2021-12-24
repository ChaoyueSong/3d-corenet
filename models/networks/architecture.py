import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import ElaIN

class ElaINResnetBlock(nn.Module):
    def __init__(self, fin, fout, ic):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv1d(fin, fmiddle, kernel_size=1)
        self.conv_1 = nn.Conv1d(fmiddle, fout, kernel_size=1)
        self.conv_s = nn.Conv1d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = ElaIN(fin, ic)
        self.norm_1 = ElaIN(fmiddle, ic)
        self.norm_s = ElaIN(fin, ic)

    def forward(self, x, addition):
        x_s = self.conv_s(self.actvn(self.norm_s(x, addition)))
        dx = self.conv_0(self.actvn(self.norm_0(x, addition)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, addition)))
        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

