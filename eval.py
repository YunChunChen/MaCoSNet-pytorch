from __future__ import print_function, division

import sys

import torch
import torch.nn as nn

from util.util import init_model
from util.util import init_test_data

from util.eval_util import compute_metric
from util.torch_util import BatchTensorToVars

from parser.parser import ArgumentParser


args, arg_groups = ArgumentParser(mode='eval').parse()


#torch.cuda.set_device(args.gpu)
use_cuda = torch.cuda.is_available()


""" Initialize model """
model = init_model(args, arg_groups, use_cuda, mode='eval')



""" Initialize dataloader """
test_data, test_loader = init_test_data(args)

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)


model.eval()
    
stats = compute_metric(args.eval_metric, model, test_data, test_loader, batch_tnf, args)
