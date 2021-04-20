import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from reprog import *
from PIL import Image
from torchvision.transforms import *
import numpy as np
import torch
import argparse
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--SavePath", type=str, default="./result/exp1", help="path to weights file")
    parser.add_argument("--DataPath", type=str, default="./data/app_dataset_cut.npy", help="path to weights file")
    parser.add_argument("--ImagePath", type=str, default="./data", help="path to weights file")
    parser.add_argument("--ImageNum", type=str, default="source1", help="path to dataset")
    parser.add_argument("--Classifier", type=str, default="resnet18", help="[resnet18, resnet50, densenet121, inception_v3]")
    parser.add_argument("--ImageSize", type=tuple, default=(3, 224, 224), help="[(3, 224, 224)(other nets), (3, 299, 299)(v3)]")
    parser.add_argument("--BatchSize", type=int, default=16, help="size of the batches")
    parser.add_argument("--Epochs", type=int, default=20, help="size of each image dimension")
    parser.add_argument("--Theta", type=float, default=0.75, help="size of each image dimension")
    parser.add_argument("--R", type=int, default=2, help="size of each image dimension")
    cfg = parser.parse_args()

    posi_path = cfg.ImagePath + '/' + cfg.ImageNum + '/placement.npy'  # v3net need to change
    blur_path = cfg.ImagePath + '/' + cfg.ImageNum + '/maskR' + str(cfg.R) + '.npy'  # v3net need to change
    pic_path = cfg.ImagePath + '/' + cfg.ImageNum + '/pic.jpg'
    data_path = cfg.DataPath

    blur_sv = cfg.SavePath + '/' + cfg.ImageNum + '/perturbation'
    inputs_sv = cfg.SavePath + '/' + cfg.ImageNum + '/result_image.png'
    ori_sv = cfg.SavePath + '/' + cfg.ImageNum + '/origin_image.png'
    red_sv = cfg.SavePath + '/' + cfg.ImageNum + '/metrics of ep'
    if not (os.path.exists(cfg.SavePath + '/' + cfg.ImageNum)):
        os.mkdir(cfg.SavePath + '/' + cfg.ImageNum)
    # print(blur_path)

    result_net = train_and_test(config=cfg, proportion=0.01, posi_path=posi_path, blur_path=blur_path,
                                pic_path=pic_path, data_path=data_path, red_path=red_sv, pre_path=None)
    # save result
    pic = result_net.bkg.cpu().numpy()
    blur = result_net.weight.cpu().detach().numpy()
    mask = result_net.slt_pix.cpu().detach().numpy()
    sample, _ = load_data(0.01, data_path, posi_path, resize=cfg.ImageSize, theta=cfg.Theta)
    inputs = (np.tanh(pic + sample[0] + np.multiply(blur, mask)) + 1) * 0.5
    inputs_img = Image.fromarray(np.uint8(255 * inputs.transpose([1, 2, 0])), mode='RGB')
    ori_pic_img = Image.fromarray(np.uint8(255 * 0.5 * (pic + 1).transpose([1, 2, 0])), mode='RGB')

    inputs_img.save(inputs_sv)
    ori_pic_img.save(ori_sv)
    np.save(blur_sv, blur)
