import os
import glob
import pickle
import logging
# import pcl
# import torch
import time
import torch.utils.data
# import torch.nn as nn
import numpy as np
import random
import cv2
# from vstsim.grasping import grasp_info


class ChameleonTongueGraspDataset(torch.utils.data.Dataset):
    """
    generate a real-time dataset and save mask
    """
    def __init__(self,
                 dir_synthetic_msk,
                 flg_dp_bg_0=True,
                 flg_qm_grav=True,
                 flg_drz_bg_0=False,
                 sigma_noise=0.0,
                 shape_final=[240, 240]):
        """
        :param dir_synthetic_msk:
        :param path_source_data:
        :param flg_dp_bg_0:        bool    the default pixels in depth image is 0 or 1
        :param flg_qm_grav:        bool    if consider the mass center for grasp quality evaluation
        :param flg_bg_0:           bool    the default grasp direction is [0, 0, 0] or [0, 0, 1]
        :param shape_final:        int[]    shape of img
        """
        self.shape_img = 1 * shape_final
        self.dir_synthetic_msk = dir_synthetic_msk

        self.grasps_list_all = self.get_all_route(dir_synthetic_msk)
        # self.dir_sc_info_all = self.get_all_route(path_source_data)

        self.amount = 0

        self.flg_dp_bg_0 = bool(flg_dp_bg_0)
        self.flg_qm_grav = bool(flg_qm_grav)
        self.flg_drz_bg_0 = bool(flg_drz_bg_0)

        self.amount = len(self.grasps_list_all)
        self.sigma_N = 1.0 * sigma_noise
        print('Dataset has been loaded.')
        print('Num. data: ', self.amount)
        print('sigma noise: ', self.sigma_N)

        np.random.seed(int(time.time()))

    def get_all_route(self, lst_route):
        file_list = []
        for i, route in enumerate(lst_route):
            file_list += self.get_file_name(route)
        return file_list

    def get_file_name(self, file_dir):
        file_list = []
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # current path
            if root.count('/') == file_dir.count('/'):
                for name in files:
                    str = file_dir + '/' + name
                    file_list.append(str)
        file_list.sort()
        return file_list

    def generate_imgs_stack(self, ind_grasp):

        dir_msk = self.grasps_list_all[ind_grasp]
        pos_name = dir_msk.find('name')
        str_info_grasp = dir_msk[pos_name:-4]
        dir_sc_data = dir_msk[0:pos_name]

        dir_msk_load = dir_sc_data + str_info_grasp + '.png'
        dir_dp0_load = dir_sc_data + 'detail/' + str_info_grasp + '_dp0.npy'
        dir_qm_load = dir_sc_data + 'detail/' + str_info_grasp + '_qm.npy'
        dir_qm_grav_load = dir_sc_data + 'detail/' + str_info_grasp + '_qm_grav.npy'
        dir_drx_load = dir_sc_data + 'detail/' + str_info_grasp + '_drx.npy'
        dir_dry_load = dir_sc_data + 'detail/' + str_info_grasp + '_dry.npy'
        dir_drz0_load = dir_sc_data + 'detail/' + str_info_grasp + '_drz0.npy'
        dir_dgripping_load = dir_sc_data + 'detail/' + str_info_grasp + '_dgripping.npy'

        msk_final = (cv2.imread(dir_msk_load, -1) / 255).astype(np.uint8)
        dp_final_bg0 = np.load(dir_dp0_load)
        qm_final = np.load(dir_qm_load)
        qm_grav_final = np.load(dir_qm_grav_load)
        drx_final = np.load(dir_drx_load)
        dry_final = np.load(dir_dry_load)
        drz_final_bg0 = np.load(dir_drz0_load)
        d_gripping_final = np.load(dir_dgripping_load)

        # update the pixels in the foreground
        ind_fg = msk_final[:, :] > 0
        msk_final[ind_fg] = 1
        # update the pixels in the background
        dp_final_bg1 = 1.0 * dp_final_bg0
        drz_final_bg1 = 1.0 * drz_final_bg0
        ind_bg = msk_final[:, :] < 1
        dp_final_bg1[ind_bg] = 1.0
        drz_final_bg1[ind_bg] = 1.0
        d_gripping_final[ind_bg] = 0.0
        return ind_fg, ind_bg, msk_final, dp_final_bg0, dp_final_bg1, qm_final, qm_grav_final, \
               drx_final, dry_final, drz_final_bg0, drz_final_bg1, d_gripping_final


    def __len__(self):
        return self.amount

    def __getitem__(self, index):
        ind_fg, _, msk_syn, dp_syn_bg0, dp_syn_bg1, qm_syn, qm_grav_syn, drx_syn, dry_syn, \
        drz_syn_bg0, drz_syn_bg1, d_gripping_syn = self.generate_imgs_stack(index)

        if self.flg_dp_bg_0:
            dp_syn_final = dp_syn_bg0
        else:
            dp_syn_final = dp_syn_bg1
        if self.flg_qm_grav:
            qm_syn_final = qm_grav_syn
        else:
            qm_syn_final = qm_syn
        if self.flg_drz_bg_0:
            drz_syn_final = drz_syn_bg0
        else:
            drz_syn_final = drz_syn_bg1

        np.random.seed(int(index))
        # scale_noise = 1.0 + (np.random.random(self.shape_img) * 2.0 * self.sigma_N - self.sigma_N)
        # dp_syn_final[ind_fg] = dp_syn_final[ind_fg] * scale_noise[ind_fg]
        arr_noise = np.random.random(self.shape_img) * 2.0 * self.sigma_N - self.sigma_N
        dp_syn_final[ind_fg] = dp_syn_final[ind_fg] + arr_noise[ind_fg]

        msk_syn = np.copy(msk_syn[np.newaxis, :, :]).astype(np.float32)
        dp_syn_final = np.copy(dp_syn_final[np.newaxis, :, :]).astype(np.float32)
        qm_syn_final = np.copy(qm_syn_final[np.newaxis, :, :]).astype(np.float32)
        drx_syn = np.copy(drx_syn[np.newaxis, :, :]).astype(np.float32)
        dry_syn = np.copy(dry_syn[np.newaxis, :, :]).astype(np.float32)
        drz_syn_final = np.copy(drz_syn_final[np.newaxis, :, :]).astype(np.float32)
        d_gripping_syn = np.copy(d_gripping_syn[np.newaxis, :, :]).astype(np.float32)

        return msk_syn, dp_syn_final, qm_syn_final, drx_syn, dry_syn, drz_syn_final, d_gripping_syn

''''''
if __name__ == '__main__':
    home_dir = os.environ['HOME']
    path_train_msk = []
    path_train_msk.append(home_dir + "/chameleon_grasp_dataset_released/stack_l2_500_s14")
    path_train_msk.append(home_dir + "/chameleon_grasp_dataset_released/stack_l2_500_s15")

    path_test_msk = []
    path_test_msk.append(home_dir + "/chameleon_grasp_dataset_released/test_stack_l2_50_s18")
    path_test_msk.append(home_dir + "/chameleon_grasp_dataset_released/test_stack_s1_50_s20")

    a = ChameleonTongueGraspDataset(dir_synthetic_msk=path_test_msk,
                                    flg_dp_bg_0=True,
                                    flg_qm_grav=True,
                                    flg_drz_bg_0=False)
    ''''''
    print("Test function __len__()", a.__len__())
    #
    b = ChameleonTongueGraspDataset(dir_synthetic_msk=path_train_msk,
                                    flg_dp_bg_0=True,
                                    flg_qm_grav=True,
                                    flg_drz_bg_0=False)
    print("Test function __len__()", b.__len__())


    print("END.")

