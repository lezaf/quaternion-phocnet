import  torch
from    torch.nn                               import Module
from    torch.nn.parameter                     import Parameter
import  torch.nn.functional                    as     F
from    src.models.quaternion_layers           import QuaternionConv2d, QuaternionLinear
from    src.spatial_pyramid_layers.gpp         import GPP


class QuaternionPHOCNet(Module):

    def __init__(self,
                 n_out,
                 in_channels,
                 gpp_type='spp',
                 pooling_levels=3,
                 pool_type='max_pool'):

        super(QuaternionPHOCNet, self).__init__()

        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown pooling_type. Must be either \'gpp\', \'spp\' or \'tpp\'')

        self.n_out           = n_out
        self.in_channels     = in_channels
        self.gpp_type        = gpp_type
        self.pooling_levels  = pooling_levels
        self.pool_type       = pool_type

        # Network structure
        self.quat_conv1_1      = QuaternionConv2d(in_channels=self.in_channels, out_channels=30, kernel_size=3, padding=1)
        self.quat_conv1_2      = QuaternionConv2d(in_channels=30, out_channels=30, kernel_size=3, padding=1)
        self.quat_conv2_1      = QuaternionConv2d(in_channels=30, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv2_2      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv3_1      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv3_2      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv3_3      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv3_4      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv3_5      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv3_6      = QuaternionConv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.quat_conv4_1      = QuaternionConv2d(in_channels=60, out_channels=120, kernel_size=3, padding=1)
        self.quat_conv4_2      = QuaternionConv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1)
        self.quat_conv4_3      = QuaternionConv2d(in_channels=120, out_channels=120, kernel_size=3, padding=1)
        self.pooling_layer_fn  = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type)
        self.quat_fc5          = QuaternionLinear(self.pooling_layer_fn.pooling_output_size, 2048)
        self.quat_fc6          = QuaternionLinear(2048, 2048)
        self.quat_fc7          = QuaternionLinear(2048, self.n_out)

        # self.reset_parameters()


    def forward(self, input):
        y = F.relu(self.quat_conv1_1(input))
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
        y = F.relu(self.quat_fc7(y))

        return y