import torch.utils.data as data
from PIL import Image
import albumentations as A
import numpy as np
from imageio import imread
import random
import torch
import time
import cv2
from PIL import ImageFile
from .transform_list import RandomCropNumpy,EnhancedCompose,RandomColor,RandomHorizontalFlip,ArrayToTensorNumpy,Normalize,CropNumpy
from torchvision import transforms
import pdb
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

def _is_pil_image(img):
    return isinstance(img, Image.Image)


class NYUDataset(data.Dataset):
    def __init__(self,data_path="datasets/nyu_depth_v2",
                 trainfile_nyu="datasets/nyudepthv2_train_files_with_gt_dense.txt",
                 testfile_nyu="datasets/nyudepthv2_test_files_with_gt_dense.txt",
                 train=True,
                 crop_size=(416,544),
                 maxdepth=80.0,
                 depthscale=1000.0) -> None:
        super().__init__()
        self.max_depth = maxdepth
        self.depth_scale = depthscale
        self.train = train
        self.data_path = data_path
        if self.train:
            self.datafile = trainfile_nyu
        else:
            self.datafile = testfile_nyu

        with open(self.datafile,'r') as f:
            self.img_label_pair = f.readlines()

        self.basic_transform = A.Compose(
            [
               A.HorizontalFlip(),
                # A.RandomCrop(crop_size[0], crop_size[1]),
                # A.RandomBrightnessContrast(),
                # A.RandomGamma()
                # A.HueSaturationValue()
            ]
        )

        self.transformer = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),None]
            ])

    def __getitem__(self, index):
        image_name,depth_name = self.img_label_pair[index].split()
        dataset_type = "train" if self.train else "test"
        image_path = self.data_path+f"/official_splits/{dataset_type}/"+image_name
        depth_path = self.data_path+f"/official_splits/{dataset_type}/"+depth_name

        rgb = Image.open(image_path)    
        gt = Image.open(depth_path)
        rgb = np.asarray(rgb,dtype=np.float32)  
        gt = np.asarray(gt, dtype=np.float32)  

        if self.train:
            rgb,gt = self.augment_train_data(rgb,gt)

        rgb = rgb/255.0
        gt = gt/self.depth_scale
        gt = np.expand_dims(gt, axis=2)
        gt = np.clip(gt, 0, self.max_depth)

        rgb, gt = self.transformer([rgb] + [gt])

        return rgb,gt
    

    def augment_train_data(self,image,depth):
        image = self.basic_transform(image=image)["image"]
        depth = self.basic_transform(image=depth)["image"]

        return image,depth


    
    def __len__(self):
        return len(self.img_label_pair)


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                CropNumpy((args.height,args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2),brightness_mult_range=(0.75, 1.25)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                # CropNumpy((args.height,args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
