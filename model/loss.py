from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from util.torch_util import expand_dim
from geotnf.transformation import GeometricTnf, ComposedGeometricTnf
import torch.nn.functional as F
import scipy.signal
import config


class CycleLoss(nn.Module):

    def __init__(self, 
                 image_size=240,
                 transform='affine', 
                 use_cuda=True):

        super(CycleLoss, self).__init__()

        self.pointTnf = PointTnf(use_cuda=use_cuda)
        self.transform = transform

        self.coord = []
        for i in range(config.NUM_OF_COORD):
            for j in range(config.NUM_OF_COORD):
                xx = []
                xx.append(float(i) * image_size / config.NUM_OF_COORD)
                xx.append(float(j) * image_size / config.NUM_OF_COORD)
                self.coord.append(xx)
        self.coord = np.expand_dims(np.array(self.coord).transpose(), axis=0)
        self.coord = torch.from_numpy(self.coord).float()

        if use_cuda:
            self.coord = self.coord.cuda()

    def forward(self, theta_forward, theta_backward):
        batch = theta_forward.size()[0]
        b,h,w = self.coord.size()
        coord = Variable(self.coord.expand(batch, h, w))

        img_size = Variable(torch.FloatTensor([[240, 240, 1]])).cuda()

        forward_norm = PointsToUnitCoords(coord, img_size)
        forward_norm = self.pointTnf.affPointTnf(theta_forward, forward_norm)
        forward_coord = PointsToPixelCoords(forward_norm, img_size)
        
        backward_norm = PointsToUnitCoords(forward_coord, img_size)
        backward_norm = self.pointTnf.affPointTnf(theta_backward, backward_norm)
        backward_coord = PointsToPixelCoords(backward_norm, img_size)

        loss = (torch.dist(coord, backward_coord, p=2) ** 2) / (config.NUM_OF_COORD * config.NUM_OF_COORD) / batch

        return loss


class TransLoss(nn.Module):

    def __init__(self, 
                 transform='affine', 
                 use_cuda=True):

        super(TransLoss, self).__init__()

        self.pointTnf = PointTnf(use_cuda=use_cuda)
        self.transform = transform

        self.coord = []
        for i in range(config.NUM_OF_COORD):
            for j in range(config.NUM_OF_COORD):
                xx = []
                xx.append(float(i))
                xx.append(float(j))
                self.coord.append(xx)
        self.coord = np.expand_dims(np.array(self.coord).transpose(), axis=0)
        self.coord = torch.from_numpy(self.coord).float()

        if use_cuda:
            self.coord = self.coord.cuda()

    def forward(self, theta_A, theta_B, theta_C):
        batch = theta_A.size()[0]
        b,h,w = self.coord.size()
        self.coord = Variable(self.coord.expand(batch, h, w))

        img_size = Variable(torch.FloatTensor([[240, 240, 1]])).cuda()

        A_norm = PointsToUnitCoords(self.coord, img_size)
        A_norm = self.pointTnf.affPointTnf(theta_A, A_norm)
        A_coord = PointsToPixelCoords(A_norm, img_size)
        
        B_norm = PointsToUnitCoords(A_coord, img_size)
        B_norm = self.pointTnf.affPointTnf(theta_B, B_norm)
        B_coord = PointsToPixelCoords(B_norm, img_size)

        C_norm = PointsToUnitCoords(B_coord, img_size)
        C_norm = self.pointTnf.affPointTnf(theta_C, C_norm)
        C_coord = PointsToPixelCoords(C_norm, img_size)

        loss = (torch.dist(self.coord, C_coord, p=2) ** 2) / (config.NUM_OF_COORD * config.NUM_OF_COORD) / batch

        return loss


class CosegLoss(nn.Module):

    def __init__(self, 
                 threshold=config.THRESHOLD,
                 use_cuda=True):

        super(CosegLoss, self).__init__()

        self.threshold = threshold

        self.affTnf = GeometricTnf(geometric_model='affine',
                                   out_h=240,
                                   out_w=240,
                                   use_cuda=use_cuda)

        self.extractor = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(self.extractor.children())[:-1])
        for name,param in self.extractor.named_parameters():    
            param.requires_grad = False 

        if use_cuda:
            self.extractor = self.extractor.cuda()

    def forward(self, image_dict, mask_dict):

        image_A = image_dict['image_A']
        image_B = image_dict['image_B']

        mask_A = F.sigmoid(mask_dict['mask_A'])
        mask_B = F.sigmoid(mask_dict['mask_B'])

        obj_A = torch.squeeze(self.extractor(torch.mul(image_A, mask_A)))
        back_A = torch.squeeze(self.extractor(torch.mul(image_A, 1.0 - mask_A)))

        obj_B = torch.squeeze(self.extractor(torch.mul(image_B, mask_B)))
        back_B = torch.squeeze(self.extractor(torch.mul(image_B, 1.0 - mask_B)))

        batch, dim = obj_A.size()
        pos = (torch.dist(obj_A, obj_B, p=2) ** 2) / dim / batch
        neg = torch.max(0, self.threshold - ((torch.dist(obj_A, back_A, p=2) ** 2 + torch.dist(obj_B, back_B, p=2) ** 2) / dim / batch))

        loss = pos + neg

        return loss


class TaskLoss(nn.Module):
    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_cuda=True):

        super(TaskLoss, self).__init__()

        self.affTnf = GeometricTnf(geometric_model='affine',
                                   out_h=out_h,
                                   out_w=out_w,
                                   use_cuda=use_cuda)

    def forward(self, theta, mask_dict):

        aff_AB = theta['aff_AB']
        aff_BA = theta['aff_BA']

        mask_A = F.sigmoid(mask_dict['mask_A'])
        mask_B = F.sigmoid(mask_dict['mask_B'])

        batch,c,h,w = mask_A.size()

        mask_Awrp_B = self.affTnf(mask_A, aff_AB)
        mask_Bwrp_A = self.affTnf(mask_B, aff_BA)

        loss_A = (F.binary_cross_entropy(mask_A, mask_Bwrp_A) + F.binary_cross_entropy(1.0 - mask_A, 1.0 - mask_Bwrp_A)) / (h * w) / batch
        loss_B = (F.binary_cross_entropy(mask_B, mask_Awrp_B) + F.binary_cross_entropy(1.0 - mask_B, 1.0 - mask_Awrp_B)) / (h * w) / batch

        loss = (loss_A + loss_B) / 2.0

        return loss


class AffMatchScore(nn.Module):

    def __init__(self, 
                 tps_grid_size=3, 
                 tps_reg_factor=0, 
                 h_matches=15, 
                 w_matches=15, 
                 use_conv_filter=False, 
                 dilation_filter=None, 
                 use_cuda=True, 
                 seg_mask=False,
                 normalize_inlier_count=False, 
                 offset_factor=227/210):

        super(AffMatchScore, self).__init__()

        self.normalize = normalize_inlier_count

        self.seg_mask = seg_mask

        self.geometricTnf = GeometricTnf(geometric_model='affine',
                                         tps_grid_size=tps_grid_size,
                                         tps_reg_factor=tps_reg_factor,
                                         out_h=h_matches, out_w=w_matches,
                                         offset_factor = offset_factor,
                                         use_cuda=use_cuda)
        # define dilation filter
        if dilation_filter is None:
            dilation_filter = generate_binary_structure(2, 2)

        # define identity mask tensor (w, h are switched and will be permuted back later)
        mask_id = np.zeros((w_matches, h_matches, w_matches*h_matches))

        idx_list = list(range(0, mask_id.size, mask_id.shape[2] + 1))
        mask_id.reshape((-1))[idx_list] = 1
        mask_id = mask_id.swapaxes(0,1)

        # perform 2D dilation to each channel 
        if not use_conv_filter:
            if not (isinstance(dilation_filter, int) and dilation_filter == 0):
                for i in range(mask_id.shape[2]):
                    mask_id[:,:,i] = binary_dilation(mask_id[:,:,i], structure=dilation_filter).astype(mask_id.dtype)
        else:
            for i in range(mask_id.shape[2]):
                flt = np.array([[1/16,1/8,1/16],
                                [1/8, 1/4, 1/8],
                                [1/16,1/8,1/16]])
                mask_id[:,:,i] = scipy.signal.convolve2d(mask_id[:,:,i], flt, mode='same', boundary='fill', fillvalue=0)
 
        # convert to PyTorch variable
        mask_id = Variable(torch.FloatTensor(mask_id).transpose(1,2).transpose(0,1).unsqueeze(0), requires_grad=False)
        self.mask_id = mask_id
        if use_cuda:
            self.mask_id = self.mask_id.cuda();

    def forward(self, theta, matches, seg_mask=None, return_outliers=False):

        if isinstance(theta, Variable): # handle normal batch transformations
            batch_size = theta.size()[0]
            theta = theta.clone()
            mask = self.geometricTnf(expand_dim(self.mask_id, 0, batch_size), theta)

            if return_outliers:
                mask_outliers = self.geometricTnf(expand_dim(1.0-self.mask_id,0,batch_size),theta)

            if self.normalize:
                epsilon = 1e-5
                mask = torch.div(mask,
                                 torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
                if return_outliers:
                    mask_outliers = torch.div(mask_outliers,
                                              torch.sum(torch.sum(torch.sum(mask_outliers+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask_outliers))

            if self.seg_mask:
                score = torch.sum(torch.sum(torch.mul(torch.sum(torch.mul(mask,matches),1),seg_mask),2),1)
            else:
                score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)

            if return_outliers:
                score_outliers = torch.sum(torch.sum(torch.sum(torch.mul(mask_outliers,matches),3),2),1)
                return (score,score_outliers)

        elif isinstance(theta, list): # handle multiple transformations per batch item, batch is in list format (used for RANSAC)
            batch_size = len(theta)
            score = []
            for b in range(batch_size):
                sample_size=theta[b].size(0)
                s=self.forward(theta[b],expand_dim(matches[b,:,:,:].unsqueeze(0),0,sample_size))
                score.append(s)

        return score
    

class TpsMatchScore(AffMatchScore):

    def __init__(self, 
                 tps_grid_size=3,
                 tps_reg_factor=0,
                 h_matches=15,
                 w_matches=15,
                 use_conv_filter=False,
                 dilation_filter=None,
                 use_cuda=True,
                 seg_mask=False,
                 normalize_inlier_count=False,
                 offset_factor=227/210):
        
        super(TpsMatchScore, self).__init__(h_matches=h_matches,
                                            w_matches=w_matches,
                                            use_conv_filter=use_conv_filter,
                                            dilation_filter=dilation_filter,
                                            use_cuda=use_cuda,
                                            seg_mask=seg_mask,
                                            normalize_inlier_count=normalize_inlier_count,
                                            offset_factor=offset_factor)
        
        self.compGeometricTnf = ComposedGeometricTnf(tps_grid_size=tps_grid_size,
                                                     tps_reg_factor=tps_reg_factor,
                                                     out_h=h_matches,
                                                     out_w=w_matches,
                                                     offset_factor=offset_factor,
                                                     use_cuda=use_cuda)
        
    def forward(self, theta_aff, theta_aff_tps, matches, seg_mask=None, return_outliers=False):

        batch_size=theta_aff.size()[0]
        mask = self.compGeometricTnf(image_batch=expand_dim(self.mask_id,0,batch_size),
                                     theta_aff=theta_aff,
                                     theta_aff_tps=theta_aff_tps)
        if return_outliers:
             mask_outliers = self.compGeometricTnf(image_batch=expand_dim(1.0-self.mask_id,0,batch_size),
                                                   theta_aff=theta_aff,
                                                   theta_aff_tps=theta_aff_tps)
        if self.normalize:
            epsilon=1e-5
            mask = torch.div(mask,
                             torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
            if return_outliers:
                mask_outliers = torch.div(mask,
                             torch.sum(torch.sum(torch.sum(mask_outliers+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask_outliers)) 

        if self.seg_mask:
            score = torch.sum(torch.sum(torch.mul(torch.sum(torch.mul(mask,matches),1),seg_mask),2),1)
        else:
            score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)

        if return_outliers:
            score_outliers = torch.sum(torch.sum(torch.sum(torch.mul(mask_outliers,matches),3),2),1)
            return (score,score_outliers)

        return score


class GridLoss(nn.Module):

    def __init__(self, 
                 geometric_model='affine', 
                 use_cuda=True, 
                 grid_size=20):

        super(GridLoss, self).__init__()

        self.geometric_model = geometric_model

        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)

        self.N = grid_size * grid_size

        X,Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P), requires_grad=False)

        self.pointTnf = PointTnf(use_cuda=use_cuda)

        if use_cuda:
            self.P = self.P.cuda();

    def forward(self, theta, theta_GT):

        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)

        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model == 'affine':
            P_prime = self.pointTnf.affPointTnf(theta, P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)

        elif self.geometric_model == 'tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)

        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)

        return loss
