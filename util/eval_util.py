from __future__ import print_function, division
import torch
import numpy as np
import os
from geotnf.transformation import GeometricTnf
from geotnf.flow import th_sampling_grid_to_np_flow, write_flo_file
import torch.nn.functional as F
from data.pf_pascal import PFPascalVal
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords

try:
    from py_util import create_file_path
except ImportError:
    from util.py_util import create_file_path

from model.loss import TpsMatchScore


def compute_metric(metric, model, dataset, dataloader, batch_tnf, args):
    N = len(dataset)
    stats = {}
    stats['aff_tps']={}

    if metric == 'pck':  
        metrics = ['pck']
        metric_fun = pck_metric

    elif metric == 'flow':
        metrics = ['flow']
        metric_fun = flow_metrics

    for key in stats.keys():
        for metric in metrics:
            stats[key][metric] = np.zeros((N,1))

    for i, batch in enumerate(dataloader):
        batch = batch_tnf(batch)

        batch_start_idx = args.batch_size * i
        batch_end_idx = np.minimum(batch_start_idx + args.batch_size, N)

        model.eval()
        
        aff_dict, tps_dict, _ = model(batch)

        theta_aff = aff_dict['aff_AB']
        theta_aff_tps = tps_dict['tps_Awrp_B']
        
        stats = metric_fun(batch, batch_start_idx, theta_aff, theta_aff_tps, stats, args)
            
        #print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))


    if metric == 'flow':
        print('Flow files have been saved to '+ args.flow_output_dir)
        return stats

    print('\n')

    for key in stats.keys():
        print('=== Results {} ==='.format(key))
        for metric in metrics:
            if isinstance(dataset, PFPascalVal):
                N_cat = int(np.max(dataset.category))
                for c in range(N_cat):
                    cat_idx = np.nonzero(dataset.category == c+1)[0]
                    print(dataset.category_names[c].ljust(15) + ': ', '{:.2%}'.format(np.mean(stats[key][metric][cat_idx])))

            results = stats[key][metric]
            good_idx = np.flatnonzero((results!=-1) * ~np.isnan(results))
            print('Total: '+ str(results.size))
            print('Valid: '+ str(good_idx.size)) 
            filtered_results = results[good_idx]
            print(metric + ':', '{:.2%}'.format(np.mean(filtered_results)))

    print('\n')

    return stats


def pck(source_points, warped_points, L_pck, alpha=0.1):
    # compute precentage of correct keypoints
    batch_size = source_points.size(0)
    pck = torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i] = torch.mean(correct_points.float())
    return pck


def mean_dist(source_points,warped_points,L_pck):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    dist=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        dist[i]=torch.mean(torch.div(point_distance,L_pck_mat))
    return dist

def point_dist_metric(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None
    
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda,
                  tps_reg_factor=args.tps_reg_factor)

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    if do_aff:
        # do affine only
        warped_points_aff_norm = pt.affPointTnf(theta_aff,target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm,source_im_size)

    if do_tps:
        # do tps only
        warped_points_tps_norm = pt.tpsPointTnf(theta_tps,target_points_norm)
        warped_points_tps = PointsToPixelCoords(warped_points_tps_norm,source_im_size)
        
    if do_aff_tps:
        # do tps+affine
        warped_points_aff_tps_norm = pt.tpsPointTnf(theta_aff_tps,target_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta_aff,warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(warped_points_aff_tps_norm,source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)


    if do_aff:
        dist_aff = mean_dist(source_points.data, warped_points_aff.data, L_pck)
        
    if do_tps:
        dist_tps = mean_dist(source_points.data, warped_points_tps.data, L_pck)
        
    if do_aff_tps:
        dist_aff_tps = mean_dist(source_points.data, warped_points_aff_tps.data, L_pck)
        
    if do_aff:
        stats['aff']['dist'][indices] = dist_aff.unsqueeze(1).cpu().numpy()
    if do_tps:
        stats['tps']['dist'][indices] = dist_tps.unsqueeze(1).cpu().numpy()
    if do_aff_tps:
        stats['aff_tps']['dist'][indices] = dist_aff_tps.unsqueeze(1).cpu().numpy() 
        
    return stats


def pck_metric(batch, batch_start_idx, theta_aff, theta_aff_tps, stats, args, use_cuda=True):
    alpha = args.pck_alpha
    
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda,
                  tps_reg_factor=args.tps_reg_factor)

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    warped_points_aff_tps_norm = pt.tpsPointTnf(theta_aff_tps, target_points_norm)
    warped_points_aff_tps_norm = pt.affPointTnf(theta_aff, warped_points_aff_tps_norm)
    warped_points_aff_tps = PointsToPixelCoords(warped_points_aff_tps_norm, source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx, batch_start_idx + current_batch_size)

    pck_aff_tps = pck(source_points.data, warped_points_aff_tps.data, L_pck, alpha)
        
    stats['aff_tps']['pck'][indices] = pck_aff_tps.unsqueeze(1).cpu().numpy() 
        
    return stats


def flow_metrics(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    result_path=args.flow_output_dir
    
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None

    pt=PointTnf(use_cuda=use_cuda)
    
    batch_size=batch['source_im_size'].size(0)
    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b,0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b,1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b,0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b,1].data.cpu().numpy())

        grid_X,grid_Y = np.meshgrid(np.linspace(-1,1,w_tgt),np.linspace(-1,1,h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X,requires_grad=False)
        grid_Y = Variable(grid_Y,requires_grad=False)
        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1,1,-1)
        grid_Y_vec = grid_Y.view(1,1,-1)

        grid_XY_vec = torch.cat((grid_X_vec,grid_Y_vec),1)        

        def pointsToGrid (x,h_tgt=h_tgt,w_tgt=w_tgt): return x.contiguous().view(1,2,h_tgt,w_tgt).transpose(1,2).transpose(2,3)

        idx = batch_start_idx+b
                
        if do_aff:
            grid_aff = pointsToGrid(pt.affPointTnf(theta_aff[b,:].unsqueeze(0),grid_XY_vec))
            flow_aff = th_sampling_grid_to_np_flow(source_grid=grid_aff,h_src=h_src,w_src=w_src)
            flow_aff_path = os.path.join(result_path,'aff',batch['flow_path'][b])
            create_file_path(flow_aff_path)
            write_flo_file(flow_aff,flow_aff_path)
        if do_tps:
            grid_tps = pointsToGrid(pt.tpsPointTnf(theta_tps[b,:].unsqueeze(0),grid_XY_vec))
            flow_tps = th_sampling_grid_to_np_flow(source_grid=grid_tps,h_src=h_src,w_src=w_src)
            flow_tps_path = os.path.join(result_path,'tps',batch['flow_path'][b])
            create_file_path(flow_tps_path)
            write_flo_file(flow_tps,flow_tps_path)
        if do_aff_tps:
            grid_aff_tps = pointsToGrid(pt.affPointTnf(theta_aff[b,:].unsqueeze(0),pt.tpsPointTnf(theta_aff_tps[b,:].unsqueeze(0),grid_XY_vec)))
            flow_aff_tps = th_sampling_grid_to_np_flow(source_grid=grid_aff_tps,h_src=h_src,w_src=w_src)
            flow_aff_tps_path = os.path.join(result_path,'aff_tps',batch['flow_path'][b])
            create_file_path(flow_aff_tps_path)
            write_flo_file(flow_aff_tps,flow_aff_tps_path)

        idx = batch_start_idx+b
    return stats


def intersection_over_union(warped_mask,target_mask): 
    relative_part_weight = torch.sum(torch.sum(target_mask.data.gt(0.5).float(),2,True),3,True)/torch.sum(target_mask.data.gt(0.5).float())
    part_iou = torch.sum(torch.sum((warped_mask.data.gt(0.5) & target_mask.data.gt(0.5)).float(),2,True),3,True)/torch.sum(torch.sum((warped_mask.data.gt(0.5) | target_mask.data.gt(0.5)).float(),2,True),3,True)
    weighted_iou = torch.sum(torch.mul(relative_part_weight,part_iou))
    return weighted_iou


def label_transfer_accuracy(warped_mask,target_mask): 
    return torch.mean((warped_mask.data.gt(0.5) == target_mask.data.gt(0.5)).double())


def localization_error(source_mask_np, target_mask_np, flow_np):
    h_tgt, w_tgt = target_mask_np.shape[0],target_mask_np.shape[1]
    h_src, w_src = source_mask_np.shape[0],source_mask_np.shape[1]

    # initial pixel positions x1,y1 in target image
    x1, y1 = np.meshgrid(range(1,w_tgt+1), range(1,h_tgt+1))
    # sampling pixel positions x2,y2
    x2 = x1 + flow_np[:,:,0]
    y2 = y1 + flow_np[:,:,1]

    # compute in-bound coords for each image
    in_bound = (x2 >= 1) & (x2 <= w_src) & (y2 >= 1) & (y2 <= h_src)
    row,col = np.where(in_bound)
    row_1=y1[row,col].flatten().astype(np.int)-1
    col_1=x1[row,col].flatten().astype(np.int)-1
    row_2=y2[row,col].flatten().astype(np.int)-1
    col_2=x2[row,col].flatten().astype(np.int)-1

    # compute relative positions
    target_loc_x,target_loc_y = obj_ptr(target_mask_np)
    source_loc_x,source_loc_y = obj_ptr(source_mask_np)
    x1_rel=target_loc_x[row_1,col_1]
    y1_rel=target_loc_y[row_1,col_1]
    x2_rel=source_loc_x[row_2,col_2]
    y2_rel=source_loc_y[row_2,col_2]

    # compute localization error
    loc_err = np.mean(np.abs(x1_rel-x2_rel)+np.abs(y1_rel-y2_rel))
    
    return loc_err

def obj_ptr(mask):
    # computes images of normalized coordinates around bounding box
    # kept function name from DSP code
    h,w = mask.shape[0],mask.shape[1]
    y, x = np.where(mask>0.5)
    left = np.min(x);
    right = np.max(x);
    top = np.min(y);
    bottom = np.max(y);
    fg_width = right-left + 1;
    fg_height = bottom-top + 1;
    x_image,y_image = np.meshgrid(range(1,w+1), range(1,h+1));
    x_image = (x_image - left)/fg_width;
    y_image = (y_image - top)/fg_height;
    return (x_image,y_image)

