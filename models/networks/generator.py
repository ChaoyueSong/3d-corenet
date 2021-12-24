import torch
import torch.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.architecture import ElaINResnetBlock as ElaINResnetBlock

class ElaINGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64
        self.fc = nn.Conv1d(3, 16 * nf, 3, padding=1)

        self.conv1 = torch.nn.Conv1d(16 * nf, 16 * nf, 1) 
        self.conv2 = torch.nn.Conv1d(16 * nf, 8 * nf, 1) 
        self.conv3 = torch.nn.Conv1d(8 * nf, 4 * nf, 1) 
        self.conv4 = torch.nn.Conv1d(4 * nf, 3, 1) 

        self.elain_block1 = ElaINResnetBlock(16 * nf, 16 * nf, 256)
        self.elain_block2 = ElaINResnetBlock(8 * nf, 8 * nf, 256)
        self.elain_block3 = ElaINResnetBlock(4 * nf, 4 * nf, 256)


    def forward(self, identity_features, warp_out):
        x = warp_out.transpose(2,1)
        addition = identity_features

        x = self.fc(x)
        x = self.conv1(x)
        x = self.elain_block1(x, addition)
        x = self.conv2(x)
        x = self.elain_block2(x, addition)
        x = self.conv3(x)
        x = self.elain_block3(x, addition)        
        x = 2*torch.tanh(self.conv4(x))

        return x

class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
        
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        ndf = opt.ngf #64

        self.layer1 = nn.Conv1d(3, ndf, 1) 
        self.layer2 = nn.Conv1d(ndf * 1, ndf * 2, 1) 
        self.layer3 = nn.Conv1d(ndf * 2, ndf * 4, 1) 

        self.norm1 = nn.InstanceNorm1d(ndf)
        self.norm2 = nn.InstanceNorm1d(ndf * 2)
        self.norm3 = nn.InstanceNorm1d(ndf * 4)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, input):
        x1 = self.layer1(input)
        x1 = self.norm1(x1)
        x2 = self.layer2(self.actvn(x1))
        x2 = self.norm2(x2)
        x3 = self.layer3(self.actvn(x2))
        result = self.norm3(x3)

        return result
