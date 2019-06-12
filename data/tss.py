from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from geotnf.flow import read_flo_file
import cv2
import config


class TSS(Dataset):

    def __init__(self, 
                 csv_file=config.TSS_TRAIN_DATA, 
                 dataset_path=config.TSS_DIR,
                 output_size=(240,240),
                 transform=None,
                 random_crop=False):

        self.random_crop = random_crop
        self.out_h, self.out_w = output_size
        self.train_data = pd.read_csv(csv_file)
        self.img_A_names = self.train_data.iloc[:,0]
        self.img_B_names = self.train_data.iloc[:,1]
        self.flip_img_A = self.train_data.iloc[:, 3].as_matrix().astype('int')
        self.pair_category = self.train_data.iloc[:, 4].as_matrix().astype('int')
        self.dataset_path = dataset_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 

        self.hash_table = self.set_dict()

    def set_dict(self):
        hash_table = dict()
        for idx in range(len(self.train_data)):
            class_label = self.pair_category[idx]
            if class_label not in hash_table:
                hash_table[class_label] = []
            if self.img_A_names[idx] not in hash_table[class_label]:
                hash_table[class_label].append(self.img_A_names[idx])
            if self.img_B_names[idx] not in hash_table[class_label]:
                hash_table[class_label].append(self.img_B_names[idx])
        return hash_table
 
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        class_label = self.pair_category[idx]
        sampled_idx = random.randint(0, len(self.hash_table[class_label])-1)

        # get pre-processed images
        image_A, im_size_A = self.get_image(self.img_A_names[idx], self.flip_img_A[idx])
        image_B, im_size_B = self.get_image(self.img_B_names[idx])
        image_C, im_size_C = self.get_image(self.hash_table[class_label][sampled_idx])

        sample = {
            'image_A': image_A,
            'image_B': image_B,
            'image_C': image_C,
            'image_A_size': im_size_A,
            'image_B_size': im_size_B,
            'image_C_size': im_size_C,
            'set': class_label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name, flip=False):
        img_name = os.path.join(self.dataset_path, img_name)
        image = io.imread(img_name)
        
        # if grayscale convert to 3-channel image 
        if image.ndim==2:
            image=np.repeat(np.expand_dims(image,2),axis=2,repeats=3)

        # do random crop
        if self.random_crop:
            h,w,c=image.shape
            top=np.random.randint(h/4)
            bottom=int(3*h/4+np.random.randint(h/4))
            left=np.random.randint(w/4)
            right=int(3*w/4+np.random.randint(w/4))
            image = image[top:bottom,left:right,:] 

        # flip horizontally if needed
        if flip:
            image=np.flip(image,1)
            
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)
 

class TSSVal(Dataset):

    def __init__(self, 
                 csv_file=config.TSS_EVAL_DATA, 
                 dataset_path=config.TSS_DIR,
                 output_size=(240,240),
                 transform=None):

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.flow_direction = self.pairs.iloc[:, 2].as_matrix().astype('int')
        self.flip_img_A = self.pairs.iloc[:, 3].as_matrix().astype('int')
        self.pair_category = self.pairs.iloc[:, 4].as_matrix().astype('int')
        self.dataset_path = dataset_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        # get pre-processed images
        image_A, im_size_A = self.get_image(self.img_A_names[idx], self.flip_img_A[idx])
        image_B, im_size_B = self.get_image(self.img_B_names[idx])

        # get flow output path
        flow_path = self.get_GT_flow_relative_path(idx)

        sample = {
            'image_A': image_A,
            'image_B': image_B,
            'image_A_size': im_size_A,
            'image_B_size': im_size_B,
            'flow_path': flow_path,
            'set': class_label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name, flip=False):
        img_name = os.path.join(self.dataset_path, img_name)
        image = io.imread(img_name)
        
        # if grayscale convert to 3-channel image 
        if image.ndim == 2:
            image=np.repeat(np.expand_dims(image,2),axis=2,repeats=3)

        # flip horizontally if needed
        if flip:
            image = np.flip(image,1)
            
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)
    
    #def get_GT_flow(self,idx):
    #    img_folder = os.path.dirname(self.img_A_names[idx])
    #    flow_dir = self.flow_direction[idx]
    #    flow_file = 'flow'+str(flow_dir)+'.flo'
    #    flow_file_path = os.path.join(self.dataset_path, img_folder , flow_file)
    #    
    #    flow = torch.FloatTensor(read_flo_file(flow_file_path))
    #
    #    return flow
    
    def get_GT_flow_relative_path(self,idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow'+str(flow_dir)+'.flo'
        flow_file_path = os.path.join(img_folder , flow_file)
        
        return flow_file_path
        
