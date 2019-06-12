from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np

from util.util import init_model, init_model_optim
from util.util import init_train_data, init_eval_data
from util.util import save_model

from util.eval_util import compute_metric
from util.torch_util import BatchTensorToVars
from parser.parser import ArgumentParser
import config


args, arg_groups = ArgumentParser(mode='train').parse()

if not os.path.exists(args.result_model_dir):
    os.makedirs(args.result_model_dir)

torch.cuda.set_device(args.gpu)
use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


if args.match_loss:
    from model.loss import AffMatchScore, TpsMatchScore
    AffMatch = AffMatchScore(**arg_groups['loss'], seg_mask=args.seg_mask)
    TpsMatch = TpsMatchScore(use_cuda=use_cuda, **arg_groups['loss'], seg_mask=args.seg_mask)

if args.cycle_loss:
    from model.loss import CycleLoss
    Cycle = CycleLoss(use_cuda=use_cuda, transform='affine')

if args.trans_loss:
    from model.loss import TransLoss
    Trans = TransLoss(use_cuda=use_cuda, transform='affine')

if args.coseg_loss:
    from model.loss import CosegLoss
    Coseg = CosegLoss(use_cuda=use_cuda, transform='affine')

if args.task_loss:
    from model.loss import TaskLoss
    Task = TaskLoss(use_cuda=use_cuda, transform='affine')


def gen_mask(corr_dict):

    mask_AB = torch.max(corr_dict['corr_AB'], dim=1, keepdim=True)[0]
    mask_BA = torch.max(corr_dict['corr_BA'], dim=1, keepdim=True)[0]

    mask_dict = {
        'mask_AB': mask_AB,
        'mask_BA': mask_BA,
    }

    return mask_dict


def loss_match(aff_dict, tps_dict, corr_dict, seg_mask=False):

    mask_dict = { 
        'mask_AB': None, 
        'mask_BA': None, 
    }

    if seg_mask:
        mask_dict = gen_mask(corr_dict)

    """ Affine matching score """
    aff_AB = AffMatch(matches=corr_dict['corr_AB'], 
                      theta=aff_dict['aff_AB'], 
                      seg_mask=mask_dict['mask_AB'])
    aff_BA = AffMatch(matches=corr_dict['corr_BA'], 
                      theta=aff_dict['aff_BA'], 
                      seg_mask=mask_dict['mask_BA'])
    aff_match_score = (aff_AB + aff_BA) / 2.0
 
    """ TPS matching score """
    tps_AB = TpsMatch(matches=corr_dict['corr_AB'], 
                      theta_aff=aff_dict['aff_AB'], 
                      theta_aff_tps=tps_dict['tps_Awrp_B'],
                      seg_mask=mask_dict['mask_AB'])
    tps_BA = TpsMatch(matches=corr_dict['corr_BA'],
                      theta_aff=aff_dict['aff_BA'],
                      theta_aff_tps=tps_dict['tps_Bwrp_A'],
                      seg_mask=mask_dict['mask_BA'])
    tps_match_score = (tps_AB + tps_BA) / 2.0

    match_score = aff_match_score + tps_match_score
    match_loss = torch.mean(-match_score)
        
    return match_loss


def loss_cycle(aff_dict):
    cycle_AB = Cycle(aff_dict['aff_AB'], aff_dict['aff_BA'])
    cycle_BA = Cycle(aff_dict['aff_BA'], aff_dict['aff_AB'])
    cycle_loss = (cycle_AB + cycle_BA) / 2.0
    return cycle_loss


def loss_trans(aff_dict):
    trans_ABCA = Trans(aff_dict['aff_AB'], aff_dict['aff_BC'], aff_dict['aff_CA'])
    trans_ACBA = Trans(aff_dict['aff_AC'], aff_dict['aff_CB'], aff_dict['aff_BA'])
    trans_BACB = Trans(aff_dict['aff_BA'], aff_dict['aff_AC'], aff_dict['aff_CB'])
    trans_BCAB = Trans(aff_dict['aff_BC'], aff_dict['aff_CA'], aff_dict['aff_AB'])
    trans_CABC = Trans(aff_dict['aff_CA'], aff_dict['aff_AB'], aff_dict['aff_BC'])
    trans_CBAC = Trans(aff_dict['aff_CB'], aff_dict['aff_BA'], aff_dict['aff_AC'])
    trans_loss = (trans_ABCA + trans_ACBA + trans_BACB + trans_BCAB + trans_CABC + trans_CBAC) / 6.0
    return trans_loss


def loss_coseg(batch, mask_dict):
    coseg_loss = Coseg(batch, mask_dict)
    return coseg_loss


def loss_task(aff_dict, mask_dict):
    task_loss = Task(aff_dict, mask_loss)
    return task_loss


def print_loss(epoch, idx, num, loss_dict):
    print_string = 'Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, idx, num, 100. * batch_idx / num)
    if args.match_loss:
        print_string += ' match: {:.6f}'.format(loss_dict['match'])        
    if args.cycle_loss:
        print_string += ' cycle: {:.6f}'.format(loss_dict['cycle'])
    if args.trans_loss:
        print_string += ' trans: {:.6f}'.format(loss_dict['trans'])
    if args.coseg_loss:
        print_string += ' coseg: {:.6f}'.format(loss_dict['coseg'])
    if args.task_loss:
        print_string += ' task: {:.6f}'.format(loss_dict['task'])
    print(print_string)
    return

def process_epoch(epoch, model, model_opt, dataloader, batch_tnf, log_interval=100):
    
    for batch_idx, batch in enumerate(dataloader):

        batch = batch_tnf(batch)

        model_opt.zero_grad()

        loss_dict = {
            'match': 0,
            'cycle': 0,
            'trans': 0,
            'coseg': 0,
            'task': 0,
        }

        aff_dict, tps_dict, corr_dict = model(batch)

        loss = 0

        if args.match_loss:
            match_loss = loss_match(aff_dict, tps_dict, corr_dict, seg_mask=args.seg_mask)
            loss_dict['match'] += match_loss.data.cpu().numpy()
            loss += args.w_match * match_loss

        if args.cycle_loss:
            cycle_loss = loss_cycle(aff_dict)
            loss_dict['cycle'] += cycle_loss.data.cpu().numpy()
            loss += args.w_cycle * cycle_loss

        if args.trans_loss:
            trans_loss = loss_trans(aff_dict)
            loss_dict['trans'] += trans_loss.data.cpu().numpy()
            loss += args.w_trans * trans_loss

        if args.coseg_loss:
            coseg_loss = loss_coseg(aff_dict)
            loss_dict['coseg'] += coseg_loss.data.cpu().numpy()
            loss += args.w_coseg * coseg_loss

        if args.task_loss:
            task_loss = loss_task(aff_dict)
            loss_dict['task'] += task_loss.data.cpu().numpy()
            loss += args.w_task * task_loss

        loss.backward()
        model_opt.step()

        if batch_idx % log_interval == 0:
            print_loss(epoch, batch_idx, len(dataloader), loss_dict)
    return


def main():

    """ Initialize model """
    model = init_model(args, arg_groups, use_cuda)


    """ Initialize dataloader """
    train_data, train_loader = init_train_data(args)

    eval_data, eval_loader = init_eval_data(args)

    
    """ Initialize optimizer """
    model_opt = init_model_optim(args, model)

    batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

    """ Evaluate initial condition """
    eval_categories = np.array(range(20)) + 1
    eval_flag = np.zeros(len(eval_data))
    for i in range(len(eval_data)):
        eval_flag[i] = sum(eval_categories == eval_data.category[i])
    eval_idx = np.flatnonzero(eval_flag) 

    model.eval()

    eval_stats = compute_metric(args.eval_metric, model, eval_data, eval_loader, batch_tnf, args)
    best_eval_pck = np.mean(eval_stats['aff_tps'][args.eval_metric][eval_idx])


    best_epoch = 1
    """ Start training """
    for epoch in range(1, args.num_epochs+1):

        model.eval()

        process_epoch(epoch, model, model_opt, train_loader, batch_tnf)

        model.eval()

        eval_stats = compute_metric(args.eval_metric, model, eval_data, eval_loader, batch_tnf, args)
        eval_pck = np.mean(eval_stats['aff_tps'][args.eval_metric][eval_idx])

        is_best = eval_pck > best_eval_pck

        if eval_pck > best_eval_pck:
            best_eval_pck = eval_pck
            best_epoch = epoch

        print('eval: {:.3f}'.format(eval_pck), 
              'best eval: {:.3f}'.format(best_eval_pck),
              'best epoch: {}'.format(best_epoch)) 

        """ Early stopping """
        if eval_pck < (best_eval_pck - 0.05):
            break
        
        save_model(args, model, is_best)

    
if __name__ == '__main__':

    main()
