import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import datetime
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import warnings
from tqdm import tqdm
import hdf5storage as hdf5
# os.chdir('/mnt/sdb/dusongcheng/data/code/dinov3/UniFusion')

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '2'    
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']


def main(json_path=r'./options/train_unifusion_vif.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    opt['path']['models'] = os.path.join(opt['path']['models'], timestamp)
    opt['path']['images'] = os.path.join(opt['path']['images'], timestamp)
    opt['path']['log'] = os.path.join(opt['path']['log'], timestamp)
    opt['path']['options'] = os.path.join(opt['path']['options'], timestamp)
    os.makedirs(opt['path']['models'], exist_ok=True)
    os.makedirs(opt['path']['images'], exist_ok=True)
    os.makedirs(opt['path']['log'], exist_ok=True)
    os.makedirs(opt['path']['options'], exist_ok=True)
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    # init_iter_G, init_path_G = option.find_last_checkpoint('./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/10-14-20-32', net_type='G')
    # init_iter_E, init_path_E = option.find_last_checkpoint('./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/10-14-20-32', net_type='E')

    # opt['path']['pretrained_netG'] = init_path_G
    # opt['path']['pretrained_netE'] = init_path_E
    
    opt['path']['pretrained_netG'] = './Model/Infrared_Visible_Fusion/G.pth'
    opt['path']['pretrained_netE'] = './Model/Infrared_Visible_Fusion/E.pth'

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-
    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)
    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)
    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            continue

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())
    need_GT = False
    idx = 0
    for test_data in tqdm(test_loader):
        idx += 1
        image_name_ext = os.path.basename(test_data['A_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'])
        util.mkdir(img_dir)

        model.feed_data(test_data, phase='test')
        model.test()
        visuals = model.current_visuals(need_H=need_GT)
        E_img = util.tensor2uint(visuals['E'])
        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
        util.imsave(E_img, save_img_path)
        print(save_img_path)
        # print("save path:{}".format(save_img_path))
        
    # if cal:
    #     CE, NMI, QNCIE, TE, EI, Qy, Qcb, EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = cal_indicators(opt['datasets']['test']['dataroot_A'], opt['datasets']['test']['dataroot_B'], img_dir)
    #     print('CE: {CE:.4f}, NMI: {NMI:.4f}, QNCIE: {QNCIE:.4f}, TE: {TE:.4f}, EI: {EI:.4f}, Qy: {Qy:.4f}, Qcb: {Qcb:.4f}, EN: {EN:.4f}, MI: {MI:.4f}, SF: {SF:.4f}, AG: {AG:.4f}')
    #     print('SD: {SD:.4f}, CC: {CC:.4f}, SCD: {SCD:.4f}, VIF: {VIF:.4f}, MSE: {MSE:.4f}, PSNR: {PSNR:.4f}, Qabf: {Qabf:.4f}, Nabf: {Nabf:.4f}, SSIM: {SSIM:.4f}, MS_SSIM: {MS_SSIM:.4f}')
    #     logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))


if __name__ == '__main__':
    main()
