import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim

from utility_functions import *
import deeplab_resnet
from get import *




net = deeplab_resnet.Res_Deeplab_no_msc(2)
net.float()



saved_state_dict = torch.load('pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth')

for i in saved_state_dict:
    i_parts = i.split('.')
    if i_parts[1] == 'layer5':
        saved_state_dict[i] = net.state_dict()[i]
    if i_parts[1] == 'conv1':
        saved_state_dict[i] = torch.cat((saved_state_dict[i], torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)

net.load_state_dict(saved_state_dict)

# Let us make it run on multiple GPUs!
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

if torch.cuda.is_available():
    print('CUDA available')
    net.cuda()
else:
    print('CUDA not available')

print('before get')
print(list(get_1x(net)))
#print(net.module.Scale.conv1)

print('after get')

