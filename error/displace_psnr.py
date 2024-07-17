import math
import os
import warnings

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm

warnings.filterwarnings("ignore")

upscale_factor = 2


def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    res = 20 * math.log10(255 / math.sqrt(mse))
    return res


def ssim(img1, img2):
    return skimage_ssim(img1, img2, multichannel=True)


for speckle_size in range(1, 5):

    test_path = rf"..\datasets\Speckle_test\HR\{speckle_size}"
    cupic_path = rf"result\INTER_CUBIC\X{upscale_factor}\deformation"
    srgan_path = rf"result\Speckle-SRGAN\X{upscale_factor}\deformation"
    ssrgan_path = rf"result\SRGAN\X{upscale_factor}\deformation"

    sr_psnr_list = []
    cup_psnr_list = []
    srgan_psnr_list = []
    sr_ssim_list = []
    cup_ssim_list = []
    srgan_ssim_list = []

    filename_list = os.listdir(test_path)

    for dist in tqdm(filename_list):
        image_name = dist.split("_dis")[0]
        dis_file_name = image_name + "_tar.png_fftcc_icgn1_r16_deformation.txt"
        sr_dis_file_name = sr_image_name = f"out_srf_{upscale_factor}_" + dis_file_name

        true_deformation = np.load(rf'{test_path}\{dist}')
        with open(os.path.join(cupic_path, dis_file_name), encoding='utf-8') as f:
            cupic_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))
        with open(os.path.join(srgan_path, sr_dis_file_name), encoding='utf-8') as f:
            srgan_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))
        with open(os.path.join(ssrgan_path, sr_dis_file_name), encoding='utf-8') as f:
            ssrgan_dic_res = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))

        handle_res = true_deformation[40:440, 40:440].reshape(160000, 2)

        cup_psnr_list.append(psnr(handle_res, cupic_dic_res))
        srgan_psnr_list.append(psnr(handle_res, srgan_dic_res))
        sr_psnr_list.append(psnr(handle_res, ssrgan_dic_res))

        cup_ssim_list.append(ssim(handle_res, cupic_dic_res))
        srgan_ssim_list.append(ssim(handle_res, srgan_dic_res))
        sr_ssim_list.append(ssim(handle_res, ssrgan_dic_res))

    print("散斑大小:" + str(speckle_size))
    print("三次线性插值:")
    print("平均psnr：" + str(np.mean(cup_psnr_list)))
    print("平均ssim：" + str(np.mean(cup_ssim_list)))
    print("SRGAN超分辨率:")
    print("平均psnr：" + str(np.mean(srgan_psnr_list)))
    print("平均ssim：" + str(np.mean(srgan_ssim_list)))
    print("散斑SRGAN超分辨率:")
    print("平均psnr：" + str(np.mean(sr_psnr_list)))
    print("平均ssim：" + str(np.mean(sr_ssim_list)))
