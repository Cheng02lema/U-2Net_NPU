
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import numpy as np
from PIL import Image
import glob




# NPU改动 support - 添加NPU支持
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    print("Warning: torch_npu not found. Please install torch_npu for NPU support.")
    torch_npu = None
# NPU改动 support - 添加NPU支持




from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET # full size version 173.6 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')
# NPU改动 support - 添加NPU支持
def is_npu_available():
    """检查NPU是否可用"""
    if torch_npu is not None:
        try:
            return torch.npu.is_available()
        except:
            return False
    return False
# NPU改动 support - 添加NPU支持

# NPU改动 support - 添加NPU支持
def get_device():
    """获取可用的设备"""
    if is_npu_available():
        return torch.device('npu:0')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
# NPU改动 support - 添加NPU支持

def main():
    # --------- 1. get image path and name ---------
    model_name='u2net'
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images' + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name+'_human_seg', model_name + '_human_seg.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)
    
    # 获取设备信息
    device = get_device()
    print(f"Using device: {device}")
    
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    
    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    
    # 直接加载到目标设备 - 针对NPU特殊处理
    print(f"Loading model from: {model_dir}")
    try:
        if device.type == 'npu':
            # NPU需要使用字符串格式的设备名称
            device_str = f"npu:{device.index}" if device.index is not None else "npu:0"
            checkpoint = torch.load(model_dir, map_location=device_str)
            net.load_state_dict(checkpoint)
            print(f"Model loaded directly to NPU: {device_str}")
            
            
# 如果不是NPU则对CUDA添加支持
            
        elif device.type == 'cuda':
            # CUDA使用字符串格式
            device_str = f"cuda:{device.index}" if device.index is not None else "cuda:0"
            checkpoint = torch.load(model_dir, map_location=device_str)
            net.load_state_dict(checkpoint)
            print(f"Model loaded directly to CUDA: {device_str}")
        else:
            # CPU加载
            checkpoint = torch.load(model_dir, map_location='cpu')
            net.load_state_dict(checkpoint)
            print("Model loaded to CPU")
        
        # 确保模型在正确的设备上
        net = net.to(device)
        
    except Exception as e:
        print(f"Error loading model directly to device: {e}")
        print("Falling back to CPU loading...")
        # 如果直接加载失败，回退到CPU加载
        try:
            checkpoint = torch.load(model_dir, map_location='cpu')
            net.load_state_dict(checkpoint)
            net = net.to(device)
            print(f"Model loaded via CPU fallback to: {device}")
        except Exception as fallback_error:
            print(f"CPU fallback also failed: {fallback_error}")
            raise
    
    net.eval()
    
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        
        # 将输入数据移动到相应设备
        inputs_test = Variable(inputs_test.to(device))
        
        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        
        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)
        
        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()