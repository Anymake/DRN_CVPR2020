# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from external.DCNv2.modules.modulated_deform_conv import ModulatedDeformConv
from external.DCNv2.functions.modulated_deform_conv_func import ModulatedDeformConvFunction



class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class RotationConvLayer(ModulatedDeformConv):
    '''Rotation convolution layer

    Construct rotation convolution layer, which is based on deformable convolution.
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1,deformable_groups=1,
                 im2col_step=64, bias=True):
        super(RotationConvLayer, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups,im2col_step,bias)
        channels_ = self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_mask = nn.Conv2d(self.in_channels, channels_,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    bias=True)
        self.init_mask()

    def init_mask(self):
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()

    def gene_offset(self, b, h, w, angle):
        """ Obtain the offset tensor for dcn module in accordance with angle tensor.

        Take a 3x3 kernel case, the offset tensor for one location is:
        off = [x0, y0, x1, y1, x1,y1, ..., x8,y8].
        For a conventional convolution, off = [0,0,0,0,...]
        The regular grid receptive field R for each position is:
        R = [(-1,-1), (-1,0), (-1,1), ..., (1,0),(1,1)]
        With the predicted angle, we first obtain the rotation matrix M:
           --                --              |
        M=|cos\theta  sin\theta|           --|------->y
          |-sin\theta cos\theta|             |
           --                --              Vx
        After otatiron, the offset tensor offset OFF_M:
        OFF_M = M * R - R
              = (M - I)*R

        :param b: The batchsize of input tensor.
        :param h: The height of feature.
        :param w: The width of feature.
        :param angle: The predict angle tensor for each object (at every location).
        :return: The offset tensor used in dcn module.
        """
        x_v = (self.kernel_size[0]-1)//2
        y_v = (self.kernel_size[1]-1)//2
        x_axis = torch.arange(-x_v, x_v+1)
        y_axis = torch.arange(-y_v, y_v+1)
        x_coor, y_coor = torch.meshgrid(x_axis, y_axis)
        x_coor = x_coor.float().contiguous().view(-1, 1)
        y_coor = y_coor.float().contiguous().view(-1, 1)
        coor = torch.cat((x_coor, y_coor), dim=1).unsqueeze(2).cuda()
        oH = (h + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        oW = (w + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        sH = self.kernel_size[0] // 2 - self.padding[0]
        sW = self.kernel_size[1] // 2 - self.padding[1]
        angle = angle[:, :, sH:sH+oH, sW:sW+oW]
        cos_theta = torch.cos(angle).unsqueeze(-1)
        # cos_theta = cos_theta[:,:,sH:sH+oH, sW:sW+oW]
        sin_theta = torch.sin(angle).unsqueeze(-1)
        # sin_theta = sin_theta[:, :, sH:sH + oH, sW:sW + oW]
        rot_theta = torch.cat((cos_theta-1, sin_theta, -sin_theta, cos_theta-1), dim=-1)
        rot_theta = rot_theta.contiguous().view(-1, 1, 2, 2)
        offset = torch.matmul(rot_theta, coor).reshape(b,oH,oW,-1).permute(0,3,1,2).contiguous()
        return offset

    def forward(self, input, angle=None, offset=None, mask=None, fp16=False):
        b, _, h, w = input.size()
        if angle is None:
            angle = torch.zeros_like(input)[:,:1,:,:]
        if offset is None:
            offset = self.gene_offset(b, h, w, angle)
            offset = offset.detach()

        if mask is None:
            mask = self.conv_mask(input)
            mask = torch.sigmoid(mask)

        return ModulatedDeformConvFunction.apply(input,
                                                 offset,
                                                 mask,
                                                 self.weight,
                                                 self.bias,
                                                 self.stride,
                                                 self.padding,
                                                 self.dilation,
                                                 self.groups,
                                                 self.deformable_groups,
                                                 self.im2col_step)

# class FeatureSelectionModule(nn.Module):
#     '''Feature Selection Module
#
#     Fuse multiple information from different branches where each neurons take different receptive fields.
#     '''
#     def __init__(self, dim_in, rot=False):
#         super(FeatureSelectionModule, self).__init__()
#         self.rot = rot
#         self._init_layers(dim_in)
#
#     def make_branch_layer(self, dim_in, dim_out, kernel, padding):
#         """ Construct the structure of a branch.
#
#         Conduct feature aggregation using rotation convolution layer and obtain 1-channel attention map for
#         subsequent feature fusion.
#
#         :param dim_in: The channel of input feature.
#         :param dim_out: The channel of output feature.
#         :param kernel: THe kernel size of rotation convolution layer.
#         :param padding: The padding for convolution.
#         :return: Object of nn.Modulelist.
#         """
#         if self.rot:
#             branch_fea = RotationConvLayer(dim_in, dim_out, kernel, stride=1, padding=padding,bias=False)
#         else:
#             branch_fea = nn.Conv2d(dim_in, dim_out,kernel_size=kernel, padding=padding)
#         branch_att = nn.Sequential(nn.ReLU(inplace=True),
#                                    nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
#                                    )
#
#         return nn.ModuleList([branch_fea, branch_att])
#
#     def _init_layers(self, dim_in):
#         # branch_ker = [(3, 3)]
#         # branch_pad = [(1, 1)]
#         branch_ker = [(3, 3), (3, 1),(1,3)]
#         branch_pad = [(1, 1), (1, 0),(0,1)]
#
#         self.branches = nn.ModuleList()
#         for ker_size, pad in zip(branch_ker, branch_pad):
#             self.branches.append(self.make_branch_layer(dim_in//4, dim_in//4, ker_size, pad))
#         self.conv_cmp = convolution(1, dim_in, dim_in//4)
#         self.conv_out = convolution(3, dim_in // 4, dim_in)
#         self.pi = np.pi
#         self._init_weights()
#
#     def _init_weights(self):
#         for name, m in self.named_modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, angle=None):
#         x_cmp = self.conv_cmp(x)
#         branch_fea = []
#         branch_att = []
#         for br in self.branches:
#             br_fea = br[0](x_cmp, angle=angle)
#             branch_fea.append(br_fea)
#             br_att = br[1](torch.mean(br_fea,dim=1,keepdim=True))
#             branch_att.append(br_att)
#
#         att =F.softmax(torch.cat(tuple(branch_att), dim=1), dim=1)
#         split_att = torch.split(att, 1, dim=1)
#         br_out = sum([fea * att for att, fea in zip(split_att, branch_fea)])
#         return self.conv_out(br_out), att

class FeatureSelectionModule(nn.Module):
    '''Feature Selection Module

    Fuse multiple information from different branches where each neurons take different receptive fields.
    '''
    def __init__(self, dim_in, rot=False):
        super(FeatureSelectionModule, self).__init__()
        self.rot = rot
        self._init_layers(dim_in)

    def make_branch_layer(self, dim_in, dim_out, kernel, padding):
        """ Construct the structure of a branch.

        Conduct feature aggregation using rotation convolution layer and obtain 1-channel attention map for
        subsequent feature fusion.

        :param dim_in: The channel of input feature.
        :param dim_out: The channel of output feature.
        :param kernel: THe kernel size of rotation convolution layer.
        :param padding: The padding for convolution.
        :return: Object of nn.Modulelist.
        """
        if self.rot:
            branch_fea = RotationConvLayer(dim_in, dim_out, kernel, stride=1, padding=padding,bias=False)
        else:
            branch_fea = nn.Conv2d(dim_in, dim_out,kernel_size=kernel, padding=padding)

        return nn.ModuleList([branch_fea])

    def _init_layers(self, dim_in, red_r=4):
        # branch_ker = [(3, 3)]
        # branch_pad = [(1, 1)]
        branch_ker = [(3, 3), (3, 1),(1,3)]
        branch_pad = [(1, 1), (1, 0),(0,1)]
        self.split = int(len(branch_ker))
        self.branches = nn.ModuleList()
        for ker_size, pad in zip(branch_ker, branch_pad):
            self.branches.extend(self.make_branch_layer(dim_in//4, dim_in//4, ker_size, pad))
        self.conv_cmp = convolution(1, dim_in, dim_in//4)
        self.conv_out = convolution(3, dim_in // 4, dim_in)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            convolution(1, dim_in//4, dim_in//4 //red_r, with_bn=True),
            nn.Conv2d(dim_in//4 //red_r, dim_in//4*self.split , 1)
        )
        self.pi = np.pi
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, angle=None):
        x_cmp = self.conv_cmp(x)
        branch_fea = []
        for br in self.branches:
            br_fea = br(x_cmp, angle=angle)
            branch_fea.append(br_fea)
        att = self.att(sum(branch_fea))
        b = att.size(0)
        att = att.view(b,-1,self.split,1)

        att =F.softmax(att, dim=2)
        split_att = torch.split(att, 1, dim=2)
        br_out = sum([fea * att for att, fea in zip(split_att, branch_fea)])
        return self.conv_out(br_out), att
