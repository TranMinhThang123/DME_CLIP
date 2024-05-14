# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from utils.calculate_error import *
from datasets.datasets_list import NYUDataset
import imageio
import imageio.core.util
from path import Path
from utils import *
from utils.logger import AverageMeter
from model.monoclip import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Refine Depth-CLIP',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting 
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--other_method',type=str,default='MonoCLIP') # default='MonoCLIP'
parser.add_argument('--trainfile_nyu', type=str, default = "datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "datasets/nyudepthv2_test_files_with_gt_dense_2.txt")
parser.add_argument('--data_path', type=str, default = "datasets/nyu_depth_v2")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')
parser.add_argument("--vis_res",default="vis_res/exp_3")
# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--dataset', type=str, default = "NYU")
parser.add_argument("--checkpoint",type=str,default="save_dir/best_model/model.pt")



def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(val_loader, model, dataset = 'KITTI'):
    ##global device
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'NYU':
        # error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel','log10', 'rmse']
    
    elif dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    errors = AverageMeter(i=len(error_names))
    length = len(val_loader)
    # switch to evaluate mode
    model.eval()
    count = 0
    # max_depth=0
    for i, (rgb_data, gt_data) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()


        input_img = rgb_data
        input_img_flip = torch.flip(input_img,[3])
        with torch.no_grad():
            
            if isinstance(model,TestModel):
                output_depth = model(input_img)
                output_depth_flip = model(input_img_flip)
            else:
                _,output_depth = model(input_img)
                _,output_depth_flip = model(input_img_flip)
            output_depth_flip = torch.flip(output_depth_flip,[3])
            output_depth = 0.5 * (output_depth + output_depth_flip)
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth, crop=False)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth, interpolate=True,idx=i)

        errors.update(err_result)
        # measure elapsed time
        if i % 50 == 0:
            print('valid: {}/{} Abs Error {:.4f} ({:.4f})'.format(i, length, errors.val[0], errors.avg[0]))

    return errors.avg,error_names


def unnormalize_image(image, mean, std):
    """
    Unnormalize an image numpy array with different mean and standard deviation for each channel.

    Args:
        image (numpy.ndarray): Image numpy array.
        mean (list or numpy.ndarray): Mean for each channel.
        std (list or numpy.ndarray): Standard deviation for each channel.

    Returns:
        numpy.ndarray: Unnormalized image numpy array.
    """
    # Ensure mean and std are numpy arrays
    mean = np.array(mean)
    std = np.array(std)

    # Unnormalize each channel separately
    unnormalized_image = image.copy()
    for i in range(image.shape[-1]):
        unnormalized_image[..., i] = (unnormalized_image[..., i] * std[i]) + mean[i]

    return unnormalized_image

def main():
    args = parser.parse_args() 

    save_path = args.vis_res
    print("Using hardware: ",device)
    print("save result to: ",save_path)


    model = HighResolutionDepthCLIP("RN50",batch_size=args.batch_size)
    model.load_state_dict(torch.load(args.checkpoint))
    # print(model)

    test_dataset = NYUDataset(train=False,testfile_nyu=args.testfile_nyu)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,drop_last=False)
    
    error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel','log10', 'rmse']

    # model.eval()

    print("="*120)
    print("="*50,"START INFERENCE","="*54)
    print("="*120)
    print("\n")
    model.eval()

    for (train_rgb_img,_) in tqdm(test_dataloader,ncols=75,desc="Inferencing : "):
        
        train_rgb_img = train_rgb_img.to(device)

        if isinstance(model,TestModel):
            preds = model(train_rgb_img)
        else:
            _, preds = model(train_rgb_img) 
        
        
        train_img_list = list(train_rgb_img)
        pred_list = list(preds)

        

        for i,(pred,train_img) in enumerate(zip(pred_list,train_img_list)):
            pred = nn.functional.interpolate(pred.unsqueeze(0),(480,640),mode="bilinear",align_corners=True)
            pred = (pred).detach().clone().cpu().numpy().squeeze()

            # pred = (pred).detach().clone().cpu().numpy().astype(int).squeeze()
            save_train = train_img.detach().permute(1,2,0).clone().cpu().numpy().squeeze()
            save_train = unnormalize_image(save_train,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])*255
            save_train = save_train.astype(int)
            np.save(f"{save_path}/predict_{i}.npy",pred)  
            cv2.imwrite(f"{save_path}/image_{i}.jpg",save_train)  

      

    # scheduler.step(metrics=loss_d)
    errors, error_names = validate(test_dataloader,model,"NYU")
    print("")
    error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], errors[0:len(errors)]))
    print(' * Avg {}'.format(error_string))
    print("")
        
    print("\n"+"="*120)
    print("="*50,"END INFERENCE","="*56)
    print("="*120)

if __name__ == "__main__":
    main()



