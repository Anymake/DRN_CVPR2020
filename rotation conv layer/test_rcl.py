
import torch
import torch.nn as nn
from rotation_conv_utils import RotationConvLayer
import numpy as np


kH = 3
kW = 1
kernel = (kH,kW)
pH = 1
pW = 0
padding = (pH,pW)
iH = iW = 3
oH = (iH + 2 * pH - kH)//1 +1
oW = (iW + 2 * pW - kW)//1 +1

deformable_groups = 1
N, inC, inH, inW = 1, 1, 3, 3
outC = 1


# check_mdconv_zero_offset()
test_rcl = RotationConvLayer(1, 1, kernel, stride=1, padding=padding, bias=False).cuda()

input = torch.arange(0,iH*iW).view(1,1,iH,iW).cuda().float()
input[0,0,2,1] = 9
input[0,0,1,2] = 10

print('input')
print(input.squeeze().data.cpu().numpy())

nn.init.constant_(test_rcl.weight, 1.0)
nn.init.constant_(test_rcl.bias, 0.)
angle = torch.zeros_like(input)
# offset = [0,0,0,0,0,0,0,0,0,0]
offset = [0,0,0,0,0,0]
offset = torch.Tensor(offset).view(2*kH*kW,1)
offset = offset.expand(2*kH*kW,oH*oW).contiguous().view(-1).view(1,2*kH*kW,oH,oW).cuda()
mask = torch.ones(N,kH*kW,oH,oW).cuda()

# conventional conv
output = test_rcl(input, angle, offset, mask)
print('output')
print(output.squeeze().data.cpu().numpy())

# with rotation angle pi/2 via modify offset directly
offset1 = [1,1,0,0,-1,-1]
offset1= torch.Tensor(offset1).view(2*kH*kW,1)
offset1 = offset1.expand(2*kH*kW,oH*oW).contiguous().view(1,2*kH*kW,oH,oW).cuda()
output1 = test_rcl(input, angle, offset1, mask)
print('output_.5pi_off')
print(output1.squeeze().data.cpu().numpy())

angle1 = torch.ones_like(input)*np.pi*0.5
output_half_pi = test_rcl(input, angle=angle1, mask=mask)
print('output_.5pi')
print(output_half_pi.squeeze().data.cpu().numpy())

angle2 = torch.ones_like(input)*np.pi*1.0
output_pi = test_rcl(input, angle=angle2, mask=mask)
print('output_.pi')
print(output_pi.squeeze().data.cpu().numpy())

angle3 = torch.ones_like(input)*np.pi*1.5
output_one_half_pi = test_rcl(input, angle=angle3, mask=mask)
print('output_1.5pi')
print(output_one_half_pi.squeeze().data.cpu().numpy())

print('done.')