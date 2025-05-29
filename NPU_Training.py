import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

# NPU相关导入
import torch_npu
from torch_npu.contrib import transfer_to_npu

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 0. NPU设备设置 --------
def setup_npu():
    """设置NPU设备"""
    if torch.npu.is_available():
        # 设置默认NPU设备
        torch.npu.set_device(0)  # 使用第一个NPU设备
        print(f"Using NPU device: {torch.npu.get_device_name(0)}")
        return True
    else:
        print("NPU is not available, falling back to CPU")
        return False

# 初始化NPU
npu_available = setup_npu()

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), 
        loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss

# ------- 2. set the directory of training dataset --------

model_name = 'u2net'  # 'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = os.path.join('DUTS-TE', 'DUTS-TE-Image' + os.sep)
tra_label_dir = os.path.join('DUTS-TE', 'DUTS-TE-Mask' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 1
batch_size_train = 32
batch_size_val = 1
train_num = 4560
val_num =1140

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]
    
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    
    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))

# 根据NPU可用性调整num_workers
num_workers = 128 if npu_available else 128
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, 
                              shuffle=True, num_workers=num_workers, 
                              pin_memory=True if npu_available else False)

# ------- 3. define model --------
# define the net
if model_name == 'u2net':
    net = U2NET(3, 1)
elif model_name == 'u2netp':
    net = U2NETP(3, 1)

# 将模型移动到NPU
if npu_available:
    net = net.npu()
    print("Model moved to NPU")
else:
    print("Model running on CPU")

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), 
                      eps=1e-08, weight_decay=0)

# NPU优化器设置（如果需要）
if npu_available:
    # 某些NPU可能需要特殊的优化器设置
    # optimizer = torch_npu.optim.NpuFusedAdam(net.parameters(), lr=0.001, 
    #                                         betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    pass

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()
    
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        
        inputs, labels = data['image'], data['label']
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
        # 将数据移动到NPU或CPU
        if npu_available:
            inputs = inputs.npu(non_blocking=True)
            labels = labels.npu(non_blocking=True)
            inputs_v = Variable(inputs, requires_grad=False)
            labels_v = Variable(labels, requires_grad=False)
        else:
            inputs_v = Variable(inputs, requires_grad=False)
            labels_v = Variable(labels, requires_grad=False)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        
        # NPU内存清理
        if npu_available:
            torch.npu.empty_cache()
        
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, 
            running_loss / ite_num4val, running_tar_loss / ite_num4val))
        
        if ite_num % save_frq == 0:
            # 保存模型时将其移回CPU（如果在NPU上）
            if npu_available:
                # 将模型状态字典移动到CPU进行保存
                cpu_state_dict = {}
                for key, value in net.state_dict().items():
                    cpu_state_dict[key] = value.cpu()
                torch.save(cpu_state_dict, model_dir + model_name + 
                          "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                              ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            else:
                torch.save(net.state_dict(), model_dir + model_name + 
                          "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                              ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print("Training completed!")