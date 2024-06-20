import numpy                                  as     np
from   numpy.random                           import RandomState
import torch
from   torch.nn                               import Module
from   torch.nn.parameter                     import Parameter
import torch.nn.functional                    as     F

from   src.init_funcs.quat_kernels_init       import *


class QuaternionConv2d(Module):

    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 init_criterion='glorot',
                 weight_init='quaternion',
                 seed=None):
                 #rotation=False,
                 #quaternion_format=True,
                 #scale=False

        super(QuaternionConv2d, self).__init__()

        self.in_channels    = in_channels  // 3
        self.out_channels   = out_channels // 3
        self.kernel_size    = (kernel_size, kernel_size)
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.groups         = groups
        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.winit_func     = {'quaternion' : quaternion_init,
                               'unitary'    : unitary_init,
                               'random'     : random_init}[self.weight_init]
        # TODO: handle bias argument
        # TODO: if more arguments added, add self initialization for this argument

        self.weight_shape = (self.out_channels, self.in_channels) + (*self.kernel_size,)

        # Attention: check if .double() works properly
        self.R_weight = Parameter(torch.Tensor(*self.weight_shape).double())
        self.I_weight = Parameter(torch.Tensor(*self.weight_shape).double())
        self.J_weight = Parameter(torch.Tensor(*self.weight_shape).double())
        self.K_weight = Parameter(torch.Tensor(*self.weight_shape).double())

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).double())
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # print('R: ', self.R_weight, '\nI: ', self.I_weight, '\nJ: ', self.J_weight, '\nK:', self.K_weight)

    def reset_parameters(self):
        r, i, j, k = self.winit_func(self.in_channels,
                                     self.out_channels,
                                     self.rng,
                                     kernel_size = self.kernel_size,
                                     criterion = self.init_criterion)

        r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)

        self.R_weight.data = r.type_as(self.R_weight.data)
        self.I_weight.data = i.type_as(self.I_weight.data)
        self.J_weight.data = j.type_as(self.J_weight.data)
        self.K_weight.data = k.type_as(self.K_weight.data)

        if self.bias is not None:
            self.bias.data.zero_()


    def forward(self, input):

        i_kernels_matrix = torch.cat([ self.R_weight, -self.K_weight,  self.J_weight], dim=1)
        j_kernels_matrix = torch.cat([ self.K_weight,  self.R_weight, -self.I_weight], dim=1)
        k_kernels_matrix = torch.cat([-self.J_weight,  self.I_weight,  self.R_weight], dim=1)

        quaternion_kernels_matrix = torch.cat([i_kernels_matrix, j_kernels_matrix, k_kernels_matrix], dim=0)
        #print('input shape: ', input.shape)
        #print('quaternion_kernels_matrix shape: ', quaternion_kernels_matrix.shape)

        # Valid input shape check
        if input.dim() != 4:
            raise Exception('Input for 2d convolution must be 4-dimensional.'
                            'Found: input.dim() = ' + str(input.dim()))

        return F.conv2d(input,
                        quaternion_kernels_matrix,
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) + ')'


class QuaternionLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_criterion='glorot',
                 weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()

        self.in_features  = in_features  // 3
        self.out_features = out_features // 3

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 3).double())
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else 1337
        self.rng            = RandomState(self.seed)
        self.winit_func     = {'quaternion': quaternion_init,
                               'unitary'   : unitary_init}[self.weight_init]

        self.R_weight = Parameter(torch.Tensor(self.in_features, self.out_features).double())
        self.I_weight = Parameter(torch.Tensor(self.in_features, self.out_features).double())
        self.J_weight = Parameter(torch.Tensor(self.in_features, self.out_features).double())
        self.K_weight = Parameter(torch.Tensor(self.in_features, self.out_features).double())

        self.reset_parameters()


    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()

        r, i, j, k = self.winit_func(in_features=self.in_features,
                                     out_features=self.out_features,
                                     rng=self.rng,
                                     criterion=self.init_criterion)

        r, i, j, k  = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)

        self.R_weight.data = r.type_as(self.R_weight.data)
        self.I_weight.data = i.type_as(self.I_weight.data)
        self.J_weight.data = j.type_as(self.J_weight.data)
        self.K_weight.data = k.type_as(self.K_weight.data)


    def forward(self, input):

        i_kernels_matrix = torch.cat([ self.R_weight, -self.K_weight,  self.J_weight], dim=1)
        j_kernels_matrix = torch.cat([ self.K_weight,  self.R_weight, -self.I_weight], dim=1)
        k_kernels_matrix = torch.cat([-self.J_weight,  self.I_weight,  self.R_weight], dim=1)

        quaternion_kernels_matrix = torch.cat([i_kernels_matrix, j_kernels_matrix, k_kernels_matrix], dim=0)

        if self.bias is not None:
            return torch.addmm(self.bias, input, quaternion_kernels_matrix)
        else:
            return torch.mm(input, quaternion_kernels_matrix)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'