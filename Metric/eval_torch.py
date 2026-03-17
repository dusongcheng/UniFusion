import numpy as np
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import os
import torch
import warnings
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
import os
from Metric_torch import *

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation_one(ir_name, vi_name, f_name):
    f_img = Image.open(f_name).convert('L')
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')

    f_img_tensor = torch.tensor(np.array(f_img)).float().to(device)
    ir_img_tensor = torch.tensor(np.array(ir_img)).float().to(device)
    vi_img_tensor = torch.tensor(np.array(vi_img)).float().to(device)

    f_img_int = np.array(f_img).astype(np.int32)
    f_img_double = np.array(f_img).astype(np.float32)

    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)


    CE = CE_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    NMI = NMI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    QNCIE = QNCIE_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    TE = TE_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    EI = EI_function(f_img_tensor)
    Qy = Qy_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    Qcb = Qcb_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    EN = EN_function(f_img_tensor)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    SF = SF_function(f_img_tensor)
    SD = SD_function(f_img_tensor)
    AG = AG_function(f_img_tensor)
    PSNR = PSNR_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    MSE = MSE_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    VIF = VIF_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    CC = CC_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    SCD = SCD_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    Nabf = Nabf_function(ir_img_tensor, vi_img_tensor, f_img_tensor)
    SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
    MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)

    return CE, NMI, QNCIE, TE, EI, Qy, Qcb, EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM

def cal_indicators(a_path, b_path, f_path):
    filelist = os.listdir(a_path)
    for i in range(len(filelist)):
        CE_list = []
        NMI_list = []
        QNCIE_list = []
        TE_list = []
        EI_list = []
        Qy_list = []
        Qcb_list = []
        EN_list = []
        MI_list = []
        SF_list = []
        AG_list = []
        SD_list = []
        CC_list = []
        SCD_list = []
        VIF_list = []
        MSE_list = []
        PSNR_list = []
        Qabf_list = []
        Nabf_list = []
        SSIM_list = []
        MS_SSIM_list = []
        filename_list = ['']
        results = []
        eval_bar = tqdm(filelist[:])
        for _, item in enumerate(eval_bar):
            ir_name = os.path.join(a_path, item)
            vi_name = os.path.join(b_path, item.replace('-A', '-B'))
            f_name = os.path.join(f_path, item.replace('jpg', 'png'))

            if os.path.exists(f_name):
                CE, NMI, QNCIE, TE, EI, Qy, Qcb, EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name, f_name)
                CE_list.append(CE)
                NMI_list.append(NMI)
                QNCIE_list.append(QNCIE)
                TE_list.append(TE)
                EI_list.append(EI)
                Qy_list.append(Qy)
                Qcb_list.append(Qcb)
                EN_list.append(EN)
                MI_list.append(MI)
                SF_list.append(SF)
                AG_list.append(AG)
                SD_list.append(SD)
                CC_list.append(CC)
                SCD_list.append(SCD)
                VIF_list.append(VIF)
                MSE_list.append(MSE)
                PSNR_list.append(PSNR)
                Qabf_list.append(Qabf)
                Nabf_list.append(Nabf)
                SSIM_list.append(SSIM)
                MS_SSIM_list.append(MS_SSIM)
                filename_list.append(item)

        CE_tensor = torch.tensor(CE_list).mean().item()
        CE_list.append(CE_tensor)
        NMI_tensor = torch.tensor(NMI_list).mean().item()
        NMI_list.append(NMI_tensor)
        QNCIE_tensor = torch.tensor(QNCIE_list).mean().item()
        QNCIE_list.append(QNCIE_tensor)
        TE_tensor = torch.tensor(TE_list).mean().item()
        TE_list.append(TE_tensor)
        EI_tensor = torch.tensor(EI_list).mean().item()
        EI_list.append(EI_tensor)
        Qy_tensor = torch.tensor(Qy_list).mean().item()
        Qy_list.append(Qy_tensor)
        Qcb_tensor = torch.tensor(Qcb_list).mean().item()
        Qcb_list.append(Qcb_tensor)
        EN_tensor = torch.tensor(EN_list).mean().item()
        EN_list.append(EN_tensor)
        MI_tensor = torch.tensor(MI_list).mean().item()
        MI_list.append(MI_tensor)
        SF_tensor = torch.tensor(SF_list).mean().item()
        SF_list.append(SF_tensor)
        AG_tensor = torch.tensor(AG_list).mean().item()
        AG_list.append(AG_tensor)
        SD_tensor = torch.tensor(SD_list).mean().item()
        SD_list.append(SD_tensor)
        CC_tensor = torch.tensor(CC_list).mean().item()
        CC_list.append(CC_tensor)
        SCD_tensor = torch.tensor(SCD_list).mean().item()
        SCD_list.append(SCD_tensor)
        VIF_tensor = torch.tensor(VIF_list).mean().item()
        VIF_list.append(VIF_tensor)
        MSE_tensor = torch.tensor(MSE_list).mean().item()
        MSE_list.append(MSE_tensor)
        PSNR_tensor = torch.tensor(PSNR_list).mean().item()
        PSNR_list.append(PSNR_tensor)
        Qabf_list.append(np.mean(Qabf_list))
        Nabf_tensor = torch.tensor(Nabf_list).mean().item()
        Nabf_list.append(Nabf_tensor)
        SSIM_tensor = torch.tensor(SSIM_list).mean().item()
        SSIM_list.append(SSIM_tensor)
        MS_SSIM_tensor = torch.tensor(MS_SSIM_list).mean().item()
        MS_SSIM_list.append(MS_SSIM_tensor)
        filename_list.append('mean')
        return CE_tensor, NMI_tensor, QNCIE_tensor, TE_tensor, EI_tensor, Qy_tensor, Qcb_tensor, EN_tensor, MI_tensor, SF_tensor, AG_tensor, SD_tensor, CC_tensor, SCD_tensor, VIF_tensor, MSE_tensor, PSNR_tensor, np.mean(Qabf_list), Nabf_tensor, SSIM_tensor, MS_SSIM_tensor

path_dict = {'ivf':{'A':'Dataset/trainsets/T&R/ir','B':'Dataset/trainsets/T&R/vi','F':'Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/images/10-14-20-32/40000'},
             'mef':{'A':'Dataset/trainsets/MEFB/under','B':'Dataset/trainsets/MEFB/over','F':'Model/Multi_Exposure_Fusion/Multi_Exposure_Fusion/images/10-20-21-35/54000'},
             'mff':{'A':'Dataset/trainsets/MFIF/A_Y','B':'Dataset/trainsets/MFIF/B_Y','F':'Model/Multi_Focus_Fusion/Multi_Focus_Fusion/images/10-20-21-49/82000'},
             'mif':{'A':'Dataset/trainsets/MFIF/A_Y','B':'Dataset/trainsets/MFIF/B_Y','F':'Model/Multi_Focus_Fusion/Multi_Focus_Fusion/images/10-20-21-49/82000'},
}

if __name__ == '__main__':
    base_root = r"/mnt/sdb/dusongcheng/data/code/dinov3/SwinFusion-master_1"
    task = 'mff'
    Path_A = os.path.join(base_root, path_dict[task]['A'])
    Path_B = os.path.join(base_root, path_dict[task]['B'])
    Path_F = os.path.join(base_root, path_dict[task]['F'])
    CE, NMI, QNCIE, TE, EI, Qy, Qcb, EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = cal_indicators(Path_A, Path_B, Path_F)
    print(f'CE: {CE:.4f}, NMI: {NMI:.4f}, QNCIE: {QNCIE:.4f}, TE: {TE:.4f}, EI: {EI:.4f}, Qy: {Qy:.4f}, Qcb: {Qcb:.4f}, EN: {EN:.4f}, MI: {MI:.4f}, SF: {SF:.4f}, AG: {AG:.4f}')
    print(f'SD: {SD:.4f}, CC: {CC:.4f}, SCD: {SCD:.4f}, VIF: {VIF:.4f}, MSE: {MSE:.4f}, PSNR: {PSNR:.4f}, Qabf: {Qabf:.4f}, Nabf: {Nabf:.4f}, SSIM: {SSIM:.4f}, MS_SSIM: {MS_SSIM:.4f}')
