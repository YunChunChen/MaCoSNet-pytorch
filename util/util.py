import os
import shutil
import numpy as np

import config

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable, grad

from model.network import WeakMatchNet
from model.network import WeakCosegNet
#from model.network import MaCoSNet

from data.pf_pascal import PFPascal
from data.pf_pascal import PFPascalVal
from data.pf_willow import PFWillow
from data.tss import TSS
from data.tss import TSSVal
from data.internet import Internet
from data.internet import InternetVal

try:
    from dataloader import DataLoader
except ImportError:
    from util.dataloader import DataLoader

try:
    from normalize import NormalizeImage
except ImportError:
    from util.normalize import NormalizeImage


def init_model(args, arg_groups, use_cuda=True, mode='train'):
    if args.model_type == 'match':
        model = init_match_model(args, arg_groups, use_cuda, mode)
    elif args.model_type == 'coseg':
        model = init_coseg_model(args, arg_groups, use_cuda, mode)
    else: # joint
        model = init_joint_model(args, arg_groups, use_cuda, mode)
    return model


def init_match_model(args, arg_groups, use_cuda=True, mode='train'):

    model = WeakMatchNet(use_cuda=use_cuda,
                         **arg_groups['model'])

    if args.model:
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        for name, param in model.FeatureExtraction.state_dict().items():
            try:
                model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])
            except KeyError:
                model.FeatureExtraction.state_dict()[name].copy_(checkpoint['FeatureExtraction.' + name])
        for name, param in model.FeatureRegression.state_dict().items():
            try:
                model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
            except KeyError:
                model.FeatureRegression.state_dict()[name].copy_(checkpoint['FeatureRegression.' + name])
        for name, param in model.FeatureRegression2.state_dict().items():
            try:
                model.FeatureRegression2.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression2.' + name])
            except KeyError:
                model.FeatureRegression2.state_dict()[name].copy_(checkpoint['FeatureRegression2.' + name])

    if mode == 'train':
        for name, param in model.FeatureExtraction.named_parameters():
            param.requires_grad = False
            if args.train_fe and np.sum([name.find(x) != -1 for x in args.fe_finetune_params]):
                param.requires_grad = True
            if args.train_fe and name.find('bn') != -1 and np.sum([name.find(x) != -1 for x in args.fe_finetune_params]):
                param.requires_grad = args.train_bn

        for name, param in model.FeatureRegression.named_parameters():
            param.requires_grad = args.train_fr
            if args.train_fr and name.find('bn') != -1:
                param.requires_grad = args.train_bn

        for name, param in model.FeatureRegression2.named_parameters():
            param.requires_grad = args.train_fr
            if args.train_fr and name.find('bn') != -1:
                param.requires_grad = args.train_bn

    return model


def init_coseg_model(args, arg_groups, use_cuda=True, mode='train'):

    model = WeakCosegNet(train_fe=args.train_fe,
                         feature_extraction_cnn='resnet101',
                         use_cuda=use_cuda)

    if args.model:
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        for name, param in model.FeatureExtraction.state_dict().items():
            try:
                model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])
            except KeyError:
                model.FeatureExtraction.state_dict()[name].copy_(checkpoint['FeatureExtraction.' + name])
        for name, param in model.Decoder.state_dict().items():
            try:
                model.Decoder.state_dict()[name].copy_(checkpoint['state_dict']['Decoder.' + name])
            except KeyError:
                model.Decoder.state_dict()[name].copy_(checkpoint['Decoder.' + name])

    if mode == 'train':
        for name, param in model.FeatureExtraction.named_parameters():
            param.requires_grad = False
            if args.train_fe and np.sum([name.find(x) != -1 for x in args.fe_finetune_params]):
                param.requires_grad = True
            if args.train_fe and name.find('bn') != -1 and np.sum([name.find(x) != -1 for x in args.fe_finetune_params]):
                param.requires_grad = args.train_bn
        for name, param in model.Decoder.named_parameters():    
            param.requires_grad = True

    return model


def init_train_data(args):

    if args.training_dataset == 'pf-pascal':
        dataset = PFPascal(transform=NormalizeImage(['image_A', 'image_B', 'image_C']), random_crop=True)
    elif args.training_dataset == 'tss':
        dataset = TSS(transform=NormalizeImage(['image_A', 'image_B', 'image_C']), random_crop=True)
    else: # internet
        dataset = Internet(transform=NormalizeImage(['image_A', 'image_B', 'image_C']), random_crop=True)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return dataset, data_loader


def init_eval_data(args):

    if args.training_dataset == 'pf-pascal':
        dataset = PFPascalVal(transform=NormalizeImage(['image_A', 'image_B']), mode='eval')
    if args.training_dataset == 'tss':
        dataset = TSSVal(transform=NormalizeImage(['image_A', 'image_B']))
    else: # Internet
        dataset = InternetVal(transform=NormalizeImage(['image_A', 'image_B']))

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return dataset, data_loader


def init_test_data(args):

    if args.eval_dataset == 'pf-pascal':
        dataset = PFPascalVal(transform=NormalizeImage(['image_A', 'image_B']), mode='test')
    elif args.eval_dataset == 'pf-willow':
        dataset = PFWillow(transform=NormalizeImage(['image_A', 'image_B']))
    elif args.eval_dataset == 'tss':
        dataset = TSSVal(transform=NormalizeImage(['image_A', 'image_B']))
    else: # Internet
        dataset = InternetVal(transform=NormalizeImage(['image_A', 'image_B']))

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return dataset, data_loader


def init_model_optim(args, model):
    
    model_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    model_opt.zero_grad()

    return model_opt


def save_model(args, model, is_best):

    print('Saving model...')

    model_name = 'match_{}_cycle_{}_trans_{}_coseg_{}_task_{}.pth.tar'.format(
                 args.w_match, args.w_cycle, args.w_trans, args.w_coseg, args.w_task)

    model_path = os.path.join(args.result_model_dir, model_name)

    torch.save(model.state_dict(), model_path)

    if is_best:
        best_model_path = os.path.join(args.result_model_dir, 'best_{}'.format(model_name))
        shutil.copyfile(model_path, best_model_path)

    return
