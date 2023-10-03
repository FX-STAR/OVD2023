import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..builder import NECKS
from .fpn import FPN
from .. import builder
from .dyhead import DyHead

@NECKS.register_module()
class CBFPN_DY(FPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''
    def __init__(self, in_channels,
                 out_channels,
                 num_outs):
        super(CBFPN_DY, self).__init__(
            in_channels,
            out_channels,
            num_outs)
        self.dyhead = DyHead(in_channels=256, out_channels=256, num_blocks=4)

    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                out = self.dyhead(out)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[-1])
            out = self.dyhead(out)
            return out
