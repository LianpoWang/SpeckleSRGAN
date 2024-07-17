import math
import os
import warnings

import cv2
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm

warnings.filterwarnings("ignore")


def psnr(img1, img2):
    img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    res = 20 * math.log10(255 / math.sqrt(mse))
    return res


def ssim(img1, img2):
    img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    return skimage_ssim(img1, img2, multichannel=True)


for speckle_size in range(1, 5):
    sr_psnr_list = []
    cup_psnr_list = []
    srgan_psnr_list = []
    sr_ssim_list = []
    cup_ssim_list = []
    srgan_ssim_list = []

    test_path = rf"..\datasets\Speckle_test\HR\{speckle_size}"
    cupic_path = rf"..\result\INTER_CUBIC"
    srgan_path = rf"..\result\SRGAN"
    ssrgan_path = rf"..\result\Speckle-SRGAN"

    filename_list = os.listdir(test_path)

    for image_name in tqdm(filename_list):
        cup_image_name = image_name
        sr_image_name = f"out_srf_2_" + image_name
        # # 三次线性插值psnr
        # cup_psnr_res = psnr(os.path.join(test_path, image_name), os.path.join(cupic_path, cup_image_name))
        # cup_ssim_res = ssim(os.path.join(test_path, image_name), os.path.join(cupic_path, cup_image_name))
        # # SRGAN超分psnr
        # srgan_psnr_res = psnr(os.path.join(test_path, image_name), os.path.join(srgan_path, sr_image_name))
        # srgan_ssim_res = ssim(os.path.join(test_path, image_name), os.path.join(srgan_path, sr_image_name))
        # 散斑SRGAN超分辨率psnr
        sr_psnr_res = psnr(os.path.join(test_path, image_name), os.path.join(ssrgan_path, image_name))
        sr_ssim_res = ssim(os.path.join(test_path, image_name), os.path.join(ssrgan_path, image_name))

        # cup_psnr_list.append(cup_psnr_res)
        # cup_ssim_list.append(cup_ssim_res)
        # srgan_psnr_list.append(srgan_psnr_res)
        # srgan_ssim_list.append(srgan_ssim_res)
        sr_psnr_list.append(sr_psnr_res)
        sr_ssim_list.append(sr_ssim_res)

    # print("三次线性插值:")
    # print("平均psnr：" + str(np.mean(cup_psnr_list)))
    # print("平均ssim：" + str(np.mean(cup_ssim_list)))
    # print("SRGAN超分辨率:")
    # print("平均psnr：" + str(np.mean(srgan_psnr_list)))
    # print("平均ssim：" + str(np.mean(srgan_ssim_list)))
    print("散斑SRGAN超分辨率:")
    print("平均psnr：" + str(np.mean(sr_psnr_list)))
    print("平均ssim：" + str(np.mean(sr_ssim_list)))
