import os
import glob
import pickle

# import pcl
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from network_module import BasicGraspModule, ResidualBlock, ResCBAMBlock

import numpy as np
import random
import cv2
# from vstsim.grasping import grasp_info


# filter_sizes = [32, 16, 8, 8, 16, 32]
# kernel_sizes = [9, 5, 3, 3, 5, 9]
# strides = [3, 2, 2, 2, 2, 3]


class BIG_Net(BasicGraspModule):

    def __init__(self, channels_in=1,
                 channel_sizes=[128, 64,  32,  32],
                 kernel_sizes =[9,   3,   3,   0],
                 strides      =[2,   2,   2,   0],
                 flg_drop=False, r_drop=0.5,
                 model_name='joint_bignet'):
        super(BIG_Net, self).__init__()

        self.model_name = str(model_name)
        self.conv1_bignet = nn.Conv2d(channels_in, channel_sizes[0],
                                     kernel_size=kernel_sizes[0], stride=strides[0],
                                     padding=(kernel_sizes[0] - 1)//2)
        self.bn1_bignet = nn.BatchNorm2d(channel_sizes[0])

        self.conv2_bignet = nn.Conv2d(channel_sizes[0], channel_sizes[1],
                                     kernel_size=kernel_sizes[1], stride=strides[1],
                                     padding=(kernel_sizes[1] - 1)//2)
        self.bn2_bignet = nn.BatchNorm2d(channel_sizes[1])

        self.conv3_bignet = nn.Conv2d(channel_sizes[1], channel_sizes[2],
                                     kernel_size=kernel_sizes[2], stride=strides[2],
                                     padding=(kernel_sizes[2] - 1)//2)
        self.bn3_bignet = nn.BatchNorm2d(channel_sizes[2])

        self.res1_bignet = ResidualBlock(channel_sizes[2], channel_sizes[3])
        self.res2_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        self.res3_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        # self.res4_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        # self.res5_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        # self.res6_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        # self.res7_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        # self.res8_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])
        # self.res9_bignet = ResidualBlock(channel_sizes[3], channel_sizes[3])

        self.res1_qm = ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        self.res2_qm = ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        self.res3_qm = ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        # self.res4_qm = ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        # self.res5_qm = ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        # self.res6_qm = ResCBAMBlock(channel_sizes[3], channel_sizes[3])

        self.res1_gd= ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        self.res2_gd = ResCBAMBlock(channel_sizes[3], channel_sizes[3])
        self.res3_gd = ResCBAMBlock(channel_sizes[3], channel_sizes[3])

        self.res1_dir = ResidualBlock(channel_sizes[3], channel_sizes[3])
        self.res2_dir = ResidualBlock(channel_sizes[3], channel_sizes[3])
        self.res3_dir = ResidualBlock(channel_sizes[3], channel_sizes[3])
        #
        self.conv_t3_qm = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=kernel_sizes[2],
                                             stride=strides[2], padding=(kernel_sizes[2] - 1) // 2, output_padding=1)
        self.bn_t3_qm = nn.BatchNorm2d(channel_sizes[2])
        self.conv_t2_qm = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=kernel_sizes[1],
                                             stride=strides[1], padding=(kernel_sizes[1] - 1) // 2, output_padding=1)
        self.bn_t2_qm = nn.BatchNorm2d(channel_sizes[1])
        self.conv_t1_qm = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=kernel_sizes[0],
                                             stride=strides[0], padding=(kernel_sizes[0] - 1) // 2, output_padding=1)
        self.bn_t1_qm = nn.BatchNorm2d(channel_sizes[0])
        #
        self.conv_t3_dir = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=kernel_sizes[2],
                                              stride=strides[2], padding=(kernel_sizes[2] - 1) // 2, output_padding=1)
        self.bn_t3_dir = nn.BatchNorm2d(channel_sizes[2])
        self.conv_t2_dir = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=kernel_sizes[1],
                                              stride=strides[1], padding=(kernel_sizes[1] - 1) // 2, output_padding=1)
        self.bn_t2_dir = nn.BatchNorm2d(channel_sizes[1])
        self.conv_t1_dir = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=kernel_sizes[0],
                                              stride=strides[0], padding=(kernel_sizes[0] - 1) // 2, output_padding=1)
        self.bn_t1_dir = nn.BatchNorm2d(channel_sizes[0])
        #
        self.conv_t3_gd = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=kernel_sizes[2],
                                              stride=strides[2], padding=(kernel_sizes[2] - 1) // 2, output_padding=1)
        self.bn_t3_gd = nn.BatchNorm2d(channel_sizes[2])
        self.conv_t2_gd = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=kernel_sizes[1],
                                              stride=strides[1], padding=(kernel_sizes[1] - 1) // 2, output_padding=1)
        self.bn_t2_gd = nn.BatchNorm2d(channel_sizes[1])
        self.conv_t1_gd = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=kernel_sizes[0],
                                              stride=strides[0], padding=(kernel_sizes[0] - 1) // 2, output_padding=1)
        self.bn_t1_gd = nn.BatchNorm2d(channel_sizes[0])

        self.qm_output = nn.Conv2d(in_channels=channel_sizes[0], out_channels=1, kernel_size=3, padding=1)
        self.drxyz_output = nn.Conv2d(in_channels=channel_sizes[0], out_channels=3, kernel_size=3, padding=1)
        self.gd_output = nn.Conv2d(in_channels=channel_sizes[0], out_channels=1, kernel_size=3, padding=1)

        self.dropout = flg_drop
        self.dropout_qm = nn.Dropout(p=r_drop)
        self.dropout_drxyz = nn.Dropout(p=r_drop)
        self.dropout_gd = nn.Dropout(p=r_drop)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_input):
        x = F.leaky_relu(self.bn1_bignet(self.conv1_bignet(x_input)))
        x = F.leaky_relu(self.bn2_bignet(self.conv2_bignet(x)))
        x = F.leaky_relu(self.bn3_bignet(self.conv3_bignet(x)))
        # x = F.leaky_relu(self.bn4_bignet(self.conv4_bignet(x)))

        x = self.res1_bignet(x)
        x = self.res2_bignet(x)
        x = self.res3_bignet(x)

        drxyz_output = self.res1_dir(x)
        drxyz_output = self.res2_dir(drxyz_output)
        drxyz_output = self.res3_dir(drxyz_output)
        drxyz_output = F.leaky_relu(self.bn_t3_dir(self.conv_t3_dir(drxyz_output)))
        drxyz_output = F.leaky_relu(self.bn_t2_dir(self.conv_t2_dir(drxyz_output)))
        drxyz_output = F.leaky_relu(self.bn_t1_dir(self.conv_t1_dir(drxyz_output)))

        qm_output = self.res1_qm(x)
        qm_output = self.res2_qm(qm_output)
        qm_output = self.res3_qm(qm_output)
        # qm_output = self.res4_qm(qm_output)
        # qm_output = self.res5_qm(qm_output)
        # qm_output = self.res6_qm(qm_output)
        qm_output = F.leaky_relu(self.bn_t3_qm(self.conv_t3_qm(qm_output)))
        qm_output = F.leaky_relu(self.bn_t2_qm(self.conv_t2_qm(qm_output)))
        qm_output = F.leaky_relu(self.bn_t1_qm(self.conv_t1_qm(qm_output)))

        gd_output = self.res1_gd(x)
        gd_output = self.res2_gd(gd_output)
        gd_output = self.res3_gd(gd_output)
        gd_output = F.leaky_relu(self.bn_t3_gd(self.conv_t3_gd(gd_output)))
        gd_output = F.leaky_relu(self.bn_t2_gd(self.conv_t2_gd(gd_output)))
        gd_output = F.leaky_relu(self.bn_t1_gd(self.conv_t1_gd(gd_output)))
        ''''''
        if self.dropout:
            qm_output = self.qm_output(self.dropout_qm(qm_output))
            drxyz_output = self.drxyz_output(self.dropout_drxyz(drxyz_output))
            gd_output = self.gd_output(self.dropout_gd(gd_output))
        else:
            qm_output = self.qm_output(qm_output)
            drxyz_output = self.drxyz_output(drxyz_output)
            gd_output = self.gd_output(gd_output)

        qm_output = torch.sigmoid(qm_output)
        drxyz_output = F.normalize(drxyz_output, p=2, dim=1)
        gd_output = torch.sigmoid(gd_output)
        return qm_output, drxyz_output, gd_output

    # overwrite
    def compute_loss(self, x_in, y_desired, msk_in=None):
        [n, c, h, w] = x_in.shape
        num_total_pixels = float(n * h * w)

        qm_desired, drxyz_desired, gd_desired = y_desired
        qm_predict, drxyz_predict, gd_predict = self(x_in)

        cos_thea = drxyz_predict.clone().detach().requires_grad_(False) * \
                   drxyz_desired.clone().detach().requires_grad_(False)
        '''
        cos_thea = drxyz_predict.clone().detach() * drxyz_desired.clone().detach()
        '''
        cos_thea = np.array(cos_thea.cpu())
        cos_thea = np.sum(cos_thea, axis=1)
        ind_invalid = cos_thea[:, :, :] > 1.0
        cos_thea[ind_invalid] = 1.0
        ind_invalid = cos_thea[:, :, :] < -1.0
        cos_thea[ind_invalid] = -1.0

        error_deg = np.rad2deg(np.abs(np.arccos(cos_thea)))
        error_deg = error_deg[:, np.newaxis, :, :]

        # error_deg = torch.tensor(error_deg, requires_grad=False)

        if msk_in is not None:
            # tmp_msk = np.array(msk_in[0, :, :, :].cpu())
            qm_predict = qm_predict * msk_in
            qm_desired = qm_desired * msk_in
            # a = np.array(drxyz_predict[0, :, :, :].cpu().detach().numpy())
            # b = np.array(drxyz_predict[0, :, :, :].cpu().detach().numpy())
            drxyz_predict = drxyz_predict * msk_in
            drxyz_desired = drxyz_desired * msk_in

            gd_predict = gd_predict * msk_in
            gd_desired = gd_desired * msk_in

            qm_loss = F.mse_loss(qm_predict, qm_desired)
            gd_loss = F.mse_loss(gd_predict, gd_desired)
            drx_loss = F.mse_loss(drxyz_predict[:, 0, :, :], drxyz_desired[:, 0, :, :])
            dry_loss = F.mse_loss(drxyz_predict[:, 1, :, :], drxyz_desired[:, 1, :, :])
            drz_loss = F.mse_loss(drxyz_predict[:, 2, :, :], drxyz_desired[:, 2, :, :])
            drxyz_loss = F.mse_loss(drxyz_predict, drxyz_desired)

            ''''''
            msk_np = np.array(msk_in.cpu().detach().numpy())
            ind_fg = np.argwhere(msk_np[:, :, :, :] > 0)
            num_fg_pixels = float(ind_fg.shape[0])

            qm_loss = qm_loss * torch.tensor(num_total_pixels/num_fg_pixels).cuda()
            gd_loss = gd_loss * torch.tensor(num_total_pixels/num_fg_pixels).cuda()
            drx_loss = drx_loss * torch.tensor(num_total_pixels/num_fg_pixels).cuda()
            dry_loss = dry_loss * torch.tensor(num_total_pixels/num_fg_pixels).cuda()
            drz_loss = drz_loss * torch.tensor(num_total_pixels/num_fg_pixels).cuda()
            drxyz_loss = drxyz_loss * torch.tensor(3.0*num_total_pixels/num_fg_pixels).cuda()
            # print(drx_loss.item() + dry_loss.item() + drz_loss.item())
            # print(drxyz_loss.item())
            error_deg = error_deg * msk_np
            ave_error_deg = error_deg.sum().sum().sum().sum() / num_fg_pixels
            # ave_error_deg = np.average(ave_error_deg, axis=0)
            ave_error_deg = torch.tensor(ave_error_deg, requires_grad=False)
            pass

        else:
            qm_loss = F.mse_loss(qm_predict, qm_desired)
            gd_loss = F.mse_loss(gd_predict, gd_desired)
            drx_loss = F.mse_loss(drxyz_predict[:, 0, :, :], drxyz_desired[:, 0, :, :])
            dry_loss = F.mse_loss(drxyz_predict[:, 1, :, :], drxyz_desired[:, 1, :, :])
            drz_loss = F.mse_loss(drxyz_predict[:, 2, :, :], drxyz_desired[:, 2, :, :])
            drxyz_loss = F.mse_loss(drxyz_predict, drxyz_desired)
            ave_error_deg = error_deg.sum().sum().sum().sum() / num_total_pixels
            # ave_error_deg = np.average(ave_error_deg, axis=0)
            ave_error_deg = torch.tensor(ave_error_deg, requires_grad=False)

        return {
            'loss': 1.0 * qm_loss + 1.0 * drxyz_loss + 1.0 * gd_loss,
            'losses': {
                'qm_loss': qm_loss,
                'gd_loss': gd_loss,
                'drxyz_loss': drxyz_loss,
                'drx_loss': drx_loss,
                'dry_loss': dry_loss,
                'drz_loss': drz_loss,
                'ave_error_angle_deg': ave_error_deg,       # average error of grasp dir
            },
            'pred': {
                'qm': qm_predict,
                'gd': gd_predict,
                'drxyz': drxyz_predict,
                'drx': drxyz_predict[:, 0, :, :],
                'dry': drxyz_predict[:, 1, :, :],
                'drz': drxyz_predict[:, 2, :, :]
            }
        }

    # overwrite
    def predict(self, x_in, msk_in=None):
        qm_predict, drxyz_predict, gd_predict = self(x_in)
        if msk_in is not None:
            qm_predict = qm_predict * msk_in
            drxyz_predict = drxyz_predict * msk_in
            gd_predict = gd_predict * msk_in
        return {
            'qm': qm_predict,
            'gd': gd_predict,
            'drxyz': drxyz_predict,
            'drx': drxyz_predict[:, 0, :, :],
            'dry': drxyz_predict[:, 1, :, :],
            'drz': drxyz_predict[:, 2, :, :]
        }

    def train_qm_gd(self):
        self.disable_all_grad()
        for name_par, para in self.named_parameters():
            if "res1_qm" in name_par:
                para.requires_grad = True
            elif "res2_qm" in name_par:
                para.requires_grad = True
            elif "res3_qm" in name_par:
                para.requires_grad = True
            elif "conv_t3_qm" in name_par:
                para.requires_grad = True
            elif "conv_t2_qm" in name_par:
                para.requires_grad = True
            elif "conv_t1_qm" in name_par:
                para.requires_grad = True
            elif "qm_output" in name_par:
                para.requires_grad = True
            elif "res1_gd" in name_par:
                para.requires_grad = True
            elif "res2_gd" in name_par:
                para.requires_grad = True
            elif "res3_gd" in name_par:
                para.requires_grad = True
            elif "conv_t3_gd" in name_par:
                para.requires_grad = True
            elif "conv_t2_gd" in name_par:
                para.requires_grad = True
            elif "conv_t1_gd" in name_par:
                para.requires_grad = True
            elif "gd_output" in name_par:
                para.requires_grad = True

    def train_dir(self):
        self.disable_all_grad()
        for name_par, para in self.named_parameters():
            if "conv1_bignet" in name_par:
                para.requires_grad = True
            elif "conv2_bignet" in name_par:
                para.requires_grad = True
            elif "conv3_bignet" in name_par:
                para.requires_grad = True
            elif "res1_bignet" in name_par:
                para.requires_grad = True
            elif "res2_bignet" in name_par:
                para.requires_grad = True
            elif "res3_bignet" in name_par:
                para.requires_grad = True
            elif "res1_dir" in name_par:
                para.requires_grad = True
            elif "res2_dir" in name_par:
                para.requires_grad = True
            elif "res3_dir" in name_par:
                para.requires_grad = True
            elif "conv_t3_dir" in name_par:
                para.requires_grad = True
            elif "conv_t2_dir" in name_par:
                para.requires_grad = True
            elif "conv_t1_dir" in name_par:
                para.requires_grad = True
            elif "drxyz_output" in name_par:
                para.requires_grad = True

    def enable_all_grad(self):
        for name_par, para in self.named_parameters():
            para.requires_grad = True

    def disable_all_grad(self):
        for name_par, para in self.named_parameters():
            para.requires_grad = False

    def print_active_layers(self):
        for name_par, para in self.named_parameters():
            if para.requires_grad == True:
                print(name_par)

    def print_name_layers(self):
        for name_par, para in self.named_parameters():
            print(name_par)

    def get_parameters_qm(self):
        return list(self.res1_qm.parameters()) + \
               list(self.res2_qm.parameters()) + \
               list(self.res3_qm.parameters()) + \
               list(self.conv_t3_qm.parameters()) + \
               list(self.conv_t2_qm.parameters()) + \
               list(self.conv_t1_qm.parameters()) + \
               list(self.qm_output.parameters())

    def get_parameters_gd(self):
        return list(self.res1_gd.parameters()) + \
               list(self.res2_gd.parameters()) + \
               list(self.res3_gd.parameters()) + \
               list(self.conv_t3_gd.parameters()) + \
               list(self.conv_t2_gd.parameters()) + \
               list(self.conv_t1_gd.parameters()) + \
               list(self.gd_output.parameters())

    def get_parameters_dir(self):
        return list(self.conv1_bignet.parameters()) + \
               list(self.conv2_bignet.parameters()) + \
               list(self.conv3_bignet.parameters()) + \
               list(self.res1_bignet.parameters()) + \
               list(self.res2_bignet.parameters()) + \
               list(self.res3_bignet.parameters()) + \
               list(self.res1_dir.parameters()) + \
               list(self.res2_dir.parameters()) + \
               list(self.res3_dir.parameters()) + \
               list(self.conv_t3_dir.parameters()) + \
               list(self.conv_t2_dir.parameters()) + \
               list(self.conv_t1_dir.parameters()) + \
               list(self.drxyz_output.parameters())

    def print_training_paras(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def sum_params_dir(self):
        sum_param = 0.0
        lst_para_dir = self.get_parameters_dir()
        for i, paras in enumerate(lst_para_dir):
            sum_param += torch.sum(paras)
        return sum_param.data

    def sum_params_qm(self):
        sum_param = 0.0
        lst_para_qm = self.get_parameters_qm()
        for i, paras in enumerate(lst_para_qm):
            sum_param += torch.sum(paras)
        return sum_param.data


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    # path_grasps = home_dir + "/chameleon_grasp_dataset/003_typical"

    model = BIG_Net()
    model.eval()
    torch.set_grad_enabled(False)
    h_img = 240
    w_img = 240
    pc = np.zeros([1, 1, h_img, w_img])
    pc[0, 0, :, :] = np.random.randn(h_img, w_img)
    # pc[0, 1, :, :] = np.random.randn(h_img, w_img)
    # pc[0, 2, :, :] = np.random.randn(h_img, w_img)

    pc = torch.from_numpy(pc.astype(np.float32))

    qm, drxyz, _ = model(pc)
    # qm = model(pc)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # b = np.copy(a[0])
    # print(b)

    print("END.")

