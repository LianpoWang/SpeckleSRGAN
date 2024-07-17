import os
import warnings
from math import sqrt

import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.filterwarnings("ignore")

for speckle_size in range(1, 5):
    test_path = rf"..\datasets\Speckle_test\HR\{speckle_size}"
    # cupic_path = rf"result\INTER_CUBIC"
    # srgan_path = rf"result\SRGAN"
    ssrgan_path = rf"..\result\Speckle-SRGAN"

    cup_rmse_list = []
    srgan_rmse_list = []
    sr_rmse_list = []

    filename_list = os.listdir(test_path)

    for image in tqdm(filename_list):
        sr_image_name = f"out_srf_2_" + image

        true_image = cv2.imread(os.path.join(test_path, image), cv2.IMREAD_GRAYSCALE)
        # cupic_image = cv2.imread(os.path.join(cupic_path, image), cv2.IMREAD_GRAYSCALE)
        # srgan_image = cv2.imread(os.path.join(srgan_path, sr_image_name), cv2.IMREAD_GRAYSCALE)
        ssrgan_image = cv2.imread(os.path.join(ssrgan_path, image), cv2.IMREAD_GRAYSCALE)

        # cup_rmse_list.append(sqrt(mean_squared_error(true_image, cupic_image)))
        # srgan_rmse_list.append(sqrt(mean_squared_error(true_image, srgan_image)))
        sr_rmse_list.append(sqrt(mean_squared_error(true_image, ssrgan_image)))
    #
    # print("三次线性插值:")
    # print("平均rmse：" + str(np.mean(cup_rmse_list)))
    # print("SRGAN超分辨率:")
    # print("平均rmse：" + str(np.mean(srgan_rmse_list)))
    print("散斑SRGAN超分辨率:")
    print("平均rmse：" + str(np.mean(sr_rmse_list)))
