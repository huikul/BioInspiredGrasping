#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang 
# E-mail     : hui.zhang@kuleuven.be
# Description: 
# Date       : 05/02/2022 14:15
# File Name  : 00_network_training_rt_dataload.py
import numpy as np
import argparse
import sys
import logging
import datetime
import os
''' block ros packages when run the python script in Ubuntu terminal
try:
    # try to block the ros package
    if os.path.exists('/opt/ros/kinetic/lib/python2.7/dist-packages'):
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    if os.path.exists('/opt/ros/kinetic/lib/python3.5/dist-packages'):
        sys.path.remove('/opt/ros/kinetic/lib/python3.5/dist-packages')
    if os.path.exists('/opt/ros/kinetic/lib/python3.6/dist-packages'):
        sys.path.remove('/opt/ros/kinetic/lib/python3.6/dist-packages')
    if os.path.exists('/opt/ros/kinetic/lib/python3.7/dist-packages'):
        sys.path.remove('/opt/ros/kinetic/lib/python3.7/dist-packages')
    if os.path.exists('/opt/ros/kinetic/lib/python3.8/dist-packages'):
        sys.path.remove('/opt/ros/kinetic/lib/python3.8/dist-packages')
except RuntimeError:
    pass
'''
import cv2
import multiprocessing
import time
import torch
import json
# from vstsim.grasping import math_robot
from dataload import ChameleonTongueGraspDataset
from network import BIG_Net
from visdom import Visdom
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

home_dir = os.environ['HOME']

'''
while True:
    local_time = time.localtime(time.time())
    if local_time.tm_mon >= 3 and local_time.tm_mday >= 9 and local_time.tm_hour >= 2:
        break
    else:
        print("sleeping...")
        time.sleep(100)
'''
path_train_msk = []
path_test_msk = []
''''''
#
path_train_msk.append(home_dir + "/chameleon_grasp_dataset_released/stack_l2_500_s14")
path_train_msk.append(home_dir + "/chameleon_grasp_dataset_released/stack_l2_500_s15")

''''''
path_test_msk.append(home_dir + "/chameleon_grasp_dataset_released/test_stack_l2_50_s18")
path_test_msk.append(home_dir + "/chameleon_grasp_dataset_released/test_stack_s1_50_s20")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='BioInspiredGrasp')

"""
    These parameters should be changed in different projects
"""
port_num = 5251
# python -m visdom.server -port 5251 -env_path ~/BioInspiredGrasping/BIG_Net/data
path_root = home_dir + '/BioInspiredGrasping/BIG_Net'

parser.add_argument('--path_save', type=str, default=path_root)
args = parser.parse_args()
"""
    Default values are enough in most of cases 
"""
parser.add_argument('--model_name', type=str, default='bignet')
parser.add_argument('--sigma_noise_img', type=float, default=0.002)
parser.add_argument('--flg_dp_bg_0', type=bool, default=True)
parser.add_argument('--flg_qm_grav', type=bool, default=True)
parser.add_argument('--flg_drz_bg_0', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=15)
parser.add_argument('--lr_adam', type=float, default=0.00005)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=1)   # save model per 10 epochs
parser.add_argument('--flg_drop', default=True, type=bool)
parser.add_argument('--r_drop', default=0.5, type=float)

args = parser.parse_args()
# args.device = torch.device('cuda') if torch.cuda.is_available and args.flg_gpu else False

# logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))


def get_gpu_tem():
    shell_str = "tem_line=`nvidia-smi | grep %` && tem1=`echo $tem_line | cut -d C -f 1` " \
                "&& tem2=`echo $tem1 | cut -d % -f 2` && echo $tem2"
    result = os.popen(shell_str)
    result_str = result.read()
    tem_str = result_str.split(' ', 3)[2]
    result.close()
    return float(tem_str)

# create model
model = BIG_Net(model_name=args.model_name, flg_drop=args.flg_drop).cuda()
'''
trained_list = list(model.state_dict().keys())
for i in range(0, len(trained_list)):
    print(i, trained_list[i])
'''

viz = Visdom(env='train_bignet', port=port_num)

grasps = ChameleonTongueGraspDataset(
    dir_synthetic_msk=path_train_msk,
    flg_dp_bg_0=args.flg_dp_bg_0,
    flg_qm_grav=args.flg_qm_grav,
    flg_drz_bg_0=args.flg_drz_bg_0,
    sigma_noise=args.sigma_noise_img)

# print(grasps[0])
train_loader = torch.utils.data.DataLoader(
    grasps,
    batch_size=args.batch_size,
    shuffle=True)

test_grasps = ChameleonTongueGraspDataset(
    dir_synthetic_msk=path_test_msk,
    flg_dp_bg_0=args.flg_dp_bg_0,
    flg_qm_grav=args.flg_qm_grav,
    flg_drz_bg_0=args.flg_drz_bg_0,
    sigma_noise=args.sigma_noise_img)

test_loader = torch.utils.data.DataLoader(
    test_grasps,
    batch_size=args.batch_size,
    shuffle=True)

''''''
optimizer_adam = optim.Adam(model.parameters(), lr=args.lr_adam)
scheduler_adam = StepLR(optimizer_adam, step_size=10, gamma=0.9)

''''''
viz.line([[0., 0.]], [0.], win='TOTAL_LOSS', opts=dict(title='TOTAL_LOSS', legend=['train', 'test']))
viz.line([[0., 0.]], [0.], win='QM_ERROR', opts=dict(title='QM_ERROR', legend=['train', 'test']))
viz.line([[0., 0.]], [0.], win='DRXYZ_ERROR', opts=dict(title='DRXYZ_ERROR', legend=['train', 'test']))
viz.line([[0., 0.]], [0.], win='ANGLE_DEG_ERROR', opts=dict(title='ANGLE_DEG_ERROR', legend=['train', 'test']))
viz.line([[0., 0.]], [0.], win='GRASP_DEPTH_ERROR', opts=dict(title='GRASP_DEPTH_ERROR', legend=['train', 'test']))


def train(model, loader, epoch):
    results = {
        'loss': 0,
        'losses': {
        }
    }

    model.train()
    torch.set_grad_enabled(True)
    dataset_size = 0

    time_start = time.time()
    max_batch = float(len(grasps))/float(args.batch_size)
    for batch_idx, (msk_in, dp_in, qm_in, drx_in, dry_in, drz_in, gd_in) in enumerate(loader):
        dataset_size += qm_in.shape[0]

        # msk_in.requires_grad = False
        msk_in = msk_in.cuda()
        dp_in = dp_in.cuda()
        qm_in = qm_in.cuda()
        gd_in = gd_in.cuda()
        drxyz_in = torch.cat((drx_in, dry_in, drz_in), 1).cuda()

        model.train_dir()
        lossd = model.compute_loss(dp_in, [qm_in, drxyz_in, gd_in], msk_in)
        print("sum_params_dir: ", model.sum_params_dir())

        loss = lossd['loss']
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        model.train_qm_gd()
        lossd = model.compute_loss(dp_in, [qm_in, drxyz_in, gd_in], msk_in)
        print("sum_params_qm: ", model.sum_params_qm())
        loss = lossd['loss']
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        results['loss'] += loss.item() * float(qm_in.shape[0])
        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item() * float(qm_in.shape[0])
        time_end = time.time()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tCons. time: {:.2f}s\tRest time:{:.2f}h'.format(
            epoch, batch_idx * args.batch_size, len(loader.dataset),
            100. * batch_idx * args.batch_size / len(loader.dataset), loss.item(),
            time_end-time_start,
            (float(args.epoch + 1 - epoch) * max_batch + max_batch - float(batch_idx))*(time_end-time_start)/60.0/60.0))
        time_start = time.time()
        print("=====================================================")
        print("=====================================================")

    results['loss'] /= float(dataset_size)
    for l in results['losses']:
        results['losses'][l] /= float(dataset_size)

    return results


def test(model, loader):
    results = {
        'loss': 0,
        'losses': {
        }
    }

    model.eval()
    torch.set_grad_enabled(False)
    dataset_size = 0

    for batch_idx, (msk_in, dp_in, qm_in, drx_in, dry_in, drz_in, gd_in) in enumerate(loader):
        dataset_size += qm_in.shape[0]

        # msk_in.requires_grad = False
        msk_in = msk_in.cuda()
        dp_in = dp_in.cuda()
        qm_in = qm_in.cuda()
        gd_in = gd_in.cuda()
        drxyz_in = torch.cat((drx_in, dry_in, drz_in), 1).cuda()

        # lossd = model.compute_loss(dp_in, [qm_in, drxyz_in])
        lossd = model.compute_loss(dp_in, [qm_in, drxyz_in, gd_in], msk_in)

        loss = lossd['loss']

        results['loss'] += loss.item() * float(qm_in.shape[0])
        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item() * float(qm_in.shape[0])

    results['loss'] /= float(dataset_size)
    for l in results['losses']:
        results['losses'][l] /= float(dataset_size)

    return results


def main():
    best_ave_loss = 99999.0

    data_loss_train = np.zeros(args.epoch + 1)
    data_loss_test = np.zeros(args.epoch + 1)
    data_error_deg_train = np.zeros(args.epoch + 1)
    data_error_deg_test = np.zeros(args.epoch + 1)

    data_loss_qm_train = np.zeros(args.epoch + 1)
    data_loss_qm_test = np.zeros(args.epoch + 1)
    data_loss_drxyz_train = np.zeros(args.epoch + 1)
    data_loss_drxyz_test = np.zeros(args.epoch + 1)

    data_loss_drx_train = np.zeros(args.epoch + 1)
    data_loss_drx_test = np.zeros(args.epoch + 1)
    data_loss_dry_train = np.zeros(args.epoch + 1)
    data_loss_dry_test = np.zeros(args.epoch + 1)
    data_loss_drz_train = np.zeros(args.epoch + 1)
    data_loss_drz_test = np.zeros(args.epoch + 1)

    data_loss_gripping_depth_train = np.zeros(args.epoch + 1)
    data_loss_gripping_depth_test = np.zeros(args.epoch + 1)

    torch.backends.cudnn.benchmark = True
    for epoch in range(0, args.epoch + 1):
        results_train = train(model, train_loader, epoch)
        print('Train done, ave_loss={:.4f}'.format(results_train['loss']))

        results_test = test(model, test_loader)
        print('Test done, ave_loss={:.4f}, qm_error={:.4f}, deg_error={:.4f}, depth_error={:.4f}'.format(results_train['loss'],
                                                                                     results_test['losses']['qm_loss']**0.5,
                                                                                     results_test['losses']['ave_error_angle_deg'],
                                                                                     results_test['losses']['gd_loss']**0.5))
        ''''''
        viz.line([[results_train['loss'], results_test['loss']]], [epoch], win='TOTAL_LOSS', update='append')
        viz.line([[results_train['losses']['qm_loss'] ** 0.5, results_test['losses']['qm_loss'] ** 0.5]],
                 [epoch], win='QM_ERROR', update='append')
        viz.line([[results_train['losses']['drxyz_loss'] ** 0.5, results_test['losses']['drxyz_loss'] ** 0.5]],
                 [epoch], win='DRXYZ_ERROR', update='append')
        viz.line([[results_train['losses']['ave_error_angle_deg'], results_test['losses']['ave_error_angle_deg']]],
                 [epoch], win='ANGLE_DEG_ERROR', update='append')
        viz.line([[results_train['losses']['gd_loss'] ** 0.5, results_test['losses']['gd_loss'] ** 0.5]],
                 [epoch], win='GRASP_DEPTH_ERROR', update='append')

        data_loss_train[epoch] = 1.0 * results_train['loss']
        data_loss_test[epoch] = 1.0 * results_test['loss']

        data_error_deg_train[epoch] = 1.0 * results_train['losses']['ave_error_angle_deg']
        data_error_deg_test[epoch] = 1.0 * results_test['losses']['ave_error_angle_deg']

        data_loss_qm_train[epoch] = 1.0 * results_train['losses']['qm_loss']
        data_loss_qm_test[epoch] = 1.0 * results_test['losses']['qm_loss']

        data_loss_drxyz_train[epoch] = 1.0 * results_train['losses']['drxyz_loss']
        data_loss_drxyz_test[epoch] = 1.0 * results_test['losses']['drxyz_loss']

        data_loss_drx_train[epoch] = 1.0 * results_train['losses']['drx_loss']
        data_loss_drx_test[epoch] = 1.0 * results_test['losses']['drx_loss']

        data_loss_dry_train[epoch] = 1.0 * results_train['losses']['dry_loss']
        data_loss_dry_test[epoch] = 1.0 * results_test['losses']['dry_loss']

        data_loss_drz_train[epoch] = 1.0 * results_train['losses']['drz_loss']
        data_loss_drz_test[epoch] = 1.0 * results_test['losses']['drz_loss']

        data_loss_gripping_depth_train[epoch] = 1.0 * results_train['losses']['gd_loss']
        data_loss_gripping_depth_test[epoch] = 1.0 * results_test['losses']['gd_loss']

        np.save(args.path_save + '/data/data_loss_train.npy', data_loss_train)
        with open(args.path_save + '/data/data_loss_train.json', 'w') as f:
            json.dump(data_loss_train.tolist(), f)
        np.save(args.path_save + '/data/data_loss_test.npy', data_loss_test)
        with open(args.path_save + '/data/data_loss_test.json', 'w') as f:
            json.dump(data_loss_test.tolist(), f)

        np.save(args.path_save + '/data/data_error_deg_train.npy', data_error_deg_train)
        with open(args.path_save + '/data/data_error_deg_train.json', 'w') as f:
            json.dump(data_error_deg_train.tolist(), f)
        np.save(args.path_save + '/data/data_error_deg_test.npy', data_error_deg_test)
        with open(args.path_save + '/data/data_error_deg_test.json', 'w') as f:
            json.dump(data_error_deg_test.tolist(), f)

        np.save(args.path_save + '/data/data_loss_qm_train.npy', data_loss_qm_train)
        with open(args.path_save + '/data/data_loss_qm_train.json', 'w') as f:
            json.dump(data_loss_qm_train.tolist(), f)
        np.save(args.path_save + '/data/data_loss_qm_test.npy', data_loss_qm_test)
        with open(args.path_save + '/data/data_loss_qm_test.json', 'w') as f:
            json.dump(data_loss_qm_test.tolist(), f)

        np.save(args.path_save + '/data/data_loss_drxyz_train.npy', data_loss_drxyz_train)
        with open(args.path_save + '/data/data_loss_drxyz_train.json', 'w') as f:
            json.dump(data_loss_drxyz_train.tolist(), f)
        np.save(args.path_save + '/data/data_loss_drxyz_test.npy', data_loss_drxyz_test)
        with open(args.path_save + '/data/data_loss_drxyz_test.json', 'w') as f:
            json.dump(data_loss_drxyz_test.tolist(), f)

        np.save(args.path_save + '/data/data_loss_drx_train.npy', data_loss_drx_train)
        np.save(args.path_save + '/data/data_loss_drx_test.npy', data_loss_drx_test)

        np.save(args.path_save + '/data/data_loss_dry_train.npy', data_loss_dry_train)
        np.save(args.path_save + '/data/data_loss_dry_test.npy', data_loss_dry_test)

        np.save(args.path_save + '/data/data_loss_drz_train.npy', data_loss_drz_train)
        np.save(args.path_save + '/data/data_loss_drz_test.npy', data_loss_drz_test)

        np.save(args.path_save + '/data/data_loss_gdepth_train.npy', data_loss_gripping_depth_train)
        with open(args.path_save + '/data/data_loss_gripping_depth_train.json', 'w') as f:
            json.dump(data_loss_gripping_depth_train.tolist(), f)
        np.save(args.path_save + '/data/data_loss_gdepth_test.npy', data_loss_gripping_depth_test)
        with open(args.path_save + '/data/data_loss_gripping_depth_test.json', 'w') as f:
            json.dump(data_loss_gripping_depth_test.tolist(), f)


        # viz.save(['train_' + str(len(train_loader.dataset)) + model.model_name])
        if epoch % args.save_interval == 0:
            path = os.path.join(args.path_save + '/model/', '{}_s{}_ep{}.model'.
                                format(model.model_name, len(train_loader.dataset), epoch))
            torch.save(model, path)
            print('Save model @ {}'.format(path))
        if best_ave_loss > results_test['loss'] and epoch > int(0.5*args.epoch):
            best_ave_loss = 1.0 * results_test['loss']
            path = os.path.join(args.path_save + '/model/', '{}_best_ave_loss.model'.
                                format(model.model_name))
            torch.save(model, path)
            print('Save model @ {}'.format(path))
        ''''''
    path = os.path.join(args.path_save + '/model/', '{}_s{}_ep{}.model'.
                        format(model.model_name, len(train_loader.dataset), epoch))
    torch.save(model, path)
    print('Save model @ {}'.format(path))
    # viz.save(['train_' + str(len(train_loader.dataset)) + model.model_name])


if __name__ == "__main__":
    main()
