from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np

from model.loss import CosegLoss

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


Coseg = CosegLoss(use_cuda=use_cuda)


def loss_coseg(batch, mask_dict):
    coseg_loss = Coseg(batch, mask_dict)
    return coseg_loss


def print_loss(epoch, idx, num, loss_dict):
    print_string = 'Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, idx, num, 100. * batch_idx / num)
    print_string += ' coseg: {:.6f}'.format(loss_dict['coseg'])
    print(print_string)
    return

def process_epoch(epoch, model, model_opt, dataloader, batch_tnf, log_interval=100):
    
    for batch_idx, batch in enumerate(dataloader):

        batch = batch_tnf(batch)

        model_opt.zero_grad()

        loss_dict = {
            'coseg': 0,
        }

        mask_dict = model(batch)

        loss = 0

        coseg_loss = loss_coseg(batch, mask_dict)
        loss_dict['coseg'] += coseg_loss.data.cpu().numpy()
        loss += args.w_coseg * coseg_loss

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
    '''
    eval_categories = np.array(range(20)) + 1
    eval_flag = np.zeros(len(eval_data))
    for i in range(len(eval_data)):
        eval_flag[i] = sum(eval_categories == eval_data.category[i])
    eval_idx = np.flatnonzero(eval_flag) 

    model.eval()

    eval_stats = compute_metric(args.eval_metric, model, eval_data, eval_loader, batch_tnf, args)
    best_eval_pck = np.mean(eval_stats['aff_tps'][args.eval_metric][eval_idx])
    '''


    best_epoch = 1
    """ Start training """
    for epoch in range(1, args.num_epochs+1):

        model.eval()

        process_epoch(epoch, model, model_opt, train_loader, batch_tnf)

        '''
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
        '''
        
        save_model(args, model, is_best)

    
if __name__ == '__main__':

    main()
