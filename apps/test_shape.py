import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

import torchvision.transforms as transforms

# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    test_dataset = TrainDataset(opt, phase='test')

    projection_mode = test_dataset.projection_mode

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    print('Using Network: ', netG.name)

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    with torch.no_grad():
        set_eval()

        if not opt.no_num_eval:
            test_losses = {}
            print('calc error (test) ...')
            test_errors = calc_error(opt, netG, cuda, test_dataset, 100)
            print('eval test MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}'.format(*test_errors))
            MSE, IOU, prec, recall = test_errors
            test_losses['MSE(test)'] = MSE
            test_losses['IOU(test)'] = IOU
            test_losses['prec(test)'] = prec
            test_losses['recall(test)'] = recall

        if not opt.no_gen_mesh:
            print('generate mesh (test) ...')
            for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                # test_data = random.choice(test_dataset)
                test_data = test_dataset[0]
                save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (
                    opt.results_path, opt.name, gen_idx, test_data['name'])
                gen_mesh(opt, netG, cuda, test_data, save_path, use_octree=True)

if __name__ == '__main__':
    train(opt)