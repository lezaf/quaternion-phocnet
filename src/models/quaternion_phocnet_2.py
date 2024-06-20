import torch
import torch.nn                        as nn
import torch.nn.functional             as F

from   src.spatial_pyramid_layers.gpp  import GPP
from   src.models.quaternion_layers_2  import QuaternionConv, QuaternionLinearAutograd


class QuaternionPHOCNet(nn.Module):

    def __init__(self, n_out, in_channels=4, gpp_type='spp', pooling_levels=3, pool_type='max_pool'):
        super(QuaternionPHOCNet, self).__init__()

        # some sanity checks
        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown pooling_type. Must be either \'gpp\', \'spp\' or \'tpp\'')

        # Setup network structure
        self.quat_conv1_1 = QuaternionConv(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv1_2 = QuaternionConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv2_1 = QuaternionConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv2_2 = QuaternionConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv3_1 = QuaternionConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv3_2 = QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv3_3 = QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv3_4 = QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv3_5 = QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv3_6 = QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv4_1 = QuaternionConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv4_2 = QuaternionConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)
        self.quat_conv4_3 = QuaternionConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, rotation=False, scale=False)

        self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type)

        self.quat_fc5 = QuaternionLinearAutograd(self.pooling_layer_fn.pooling_output_size, 2048, rotation=False, scale=False)
        self.quat_fc6 = QuaternionLinearAutograd(2048, 2048, rotation=False, scale=False)
        self.quat_fc7 = QuaternionLinearAutograd(2048, n_out, rotation=False, scale=False)


    def forward(self, x):
        y = F.relu(self.quat_conv1_1(x))
        y = F.relu(self.quat_conv1_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.quat_conv2_1(y))
        y = F.relu(self.quat_conv2_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.quat_conv3_1(y))
        y = F.relu(self.quat_conv3_2(y))
        y = F.relu(self.quat_conv3_3(y))
        y = F.relu(self.quat_conv3_4(y))
        y = F.relu(self.quat_conv3_5(y))
        y = F.relu(self.quat_conv3_6(y))
        y = F.relu(self.quat_conv4_1(y))
        y = F.relu(self.quat_conv4_2(y))
        y = F.relu(self.quat_conv4_3(y))

        y = self.pooling_layer_fn.forward(y)

        y = F.relu(self.quat_fc5(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.relu(self.quat_fc6(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.quat_fc7(y)

        return y