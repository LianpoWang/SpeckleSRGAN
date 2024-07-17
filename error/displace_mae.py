import os
import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

warnings.filterwarnings("ignore")

upscale_factor = 2

for speckle_size in range(1, 5):
    test_path = rf"..\datasets\Speckle_test\HR\{speckle_size}"
    cupic_path = rf"res\INTER_CUBIC\X{upscale_factor}\deformation"
    srgan_path = rf"res\Speckle-SRGAN\X{upscale_factor}\deformation"
    ssrgan_path = rf"res\SRGAN\X{upscale_factor}\deformation"

    cup_mae_list = []
    srgan_mae_list = []
    sr_mae_list = []

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

        cup_mae_list.append(mean_absolute_error(handle_res, cupic_dic_res))
        srgan_mae_list.append(mean_absolute_error(handle_res, srgan_dic_res))
        sr_mae_list.append(mean_absolute_error(handle_res, ssrgan_dic_res))

    print("散斑大小:" + str(speckle_size))
    print("三次线性插值:")
    print("平均mae：" + str(np.mean(cup_mae_list)))
    print("SRGAN超分辨率:")
    print("平均mae：" + str(np.mean(srgan_mae_list)))
    print("散斑SRGAN超分辨率:")
    print("平均mae：" + str(np.mean(sr_mae_list)))
