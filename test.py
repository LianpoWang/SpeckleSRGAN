import argparse
import os

import torch
from PIL import Image
from torchvision.transforms import ToPILImage, transforms
from tqdm import tqdm

from model.model import Generator

filenameList = os.listdir(r'datasets/Speckle_test/LR')
filepath = r'datasets/Speckle_test/LR/'
save_path = r'result/Speckle-SRGAN'

os.makedirs(save_path, exist_ok=True)

gpu_index = 0

with torch.no_grad():
    for image_name in tqdm(filenameList):

        if os.path.splitext(image_name)[-1][1:] in ("png", "jpg", "bmp"):
            parser = argparse.ArgumentParser(description='Test Single Image')
            parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
            parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
            parser.add_argument('--image_name', default=filepath + image_name, type=str,
                                help='test low resolution image name')
            parser.add_argument('--model_name', default='netG_epoch_2_100.pth', type=str,
                                help='generator model epoch name')
            opt = parser.parse_args()

            UPSCALE_FACTOR = opt.upscale_factor
            TEST_MODE = True if opt.test_mode == 'GPU' else False
            IMAGE_NAME = opt.image_name
            MODEL_NAME = opt.model_name

            model = Generator(UPSCALE_FACTOR).eval()
            if TEST_MODE:
                model.cuda(gpu_index)
                model.load_state_dict(
                    torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cuda:' + str(gpu_index))),
                    strict=False)
            else:
                model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

            image = Image.open(IMAGE_NAME).convert('L')

            transf = transforms.ToTensor()
            img_tensor = transf(image).unsqueeze(0)

            if TEST_MODE:
                image = img_tensor.cuda(gpu_index)

            # start = time.clock()
            out = model(image)
            # elapsed = (time.clock() - start)
            # print('cost' + str(elapsed) + 's')
            out_img = ToPILImage()(out[0].data.cpu())
            out_img.save(save_path + '/' + image_name)
