import os
import argparse
from tqdm import tqdm
from rknn.api import RKNN
import numpy as np
from utils import img_read, BGR2YCrCb, YCrCb2BGR, img_save


def main(args):

    # 加载模型
    fusemodel = RKNN(verbose=False)
    fusemodel.load_rknn(args.model_path)
    fusemodel.init_runtime(
        target=args.target,
        core_mask=RKNN.NPU_CORE_ALL,
    )
    b = args.rknn_batch

    # 读取图像数据
    img_list = os.listdir(args.infrared_path)

    # 遍历融合图像数据
    for i in tqdm(range(0, len(img_list), b)):

        name_list = img_list[i:i+b]
        vi_imgs = []
        Y_list = []
        Cr_list = []
        Cb_list = []

        for img_name in name_list:

        
            ir_img_path = os.path.join(args.infrared_path, img_name)
            vi_img_path = os.path.join(args.thermal_path, img_name)

            ir_img = img_read(path=ir_img_path, is_vis=True) # (H, W, C) float32
            vi_img = img_read(path=vi_img_path, is_vis=False) # (1, 1, H, W) int8

            

            Y, Cr, Cb = BGR2YCrCb(ir_img) # Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32
            vi_imgs.append(vi_img)
            Y_list.append(Y)
            Cr_list.append(Cr)
            Cb_list.append(Cb)
        Y_con = np.concatenate(Y_list, axis=0)
        vi_con = np.concatenate(vi_imgs, axis=0)


        inputs = [Y_con, vi_con]
        fuse_imgs = fusemodel.inference(inputs=inputs, data_format='nchw')[0] # int8
        fuse_list = []
        for i in range(fuse_imgs.shape[0]):
            fuse_list.append(fuse_imgs[i][np.newaxis, ...])


        for fuse_img, Cr, Cb, img_name in zip(fuse_list, Cr_list, Cb_list, img_list):
            fuse_img_process = YCrCb2BGR(fuse_img, Cr, Cb) # (H, W, C) float32
            save_path = os.path.join(args.results_path, img_name)
            img_save(fuse_img_process, save_path)

    # 释放资源
    fusemodel.release()


def parse_options():
    parser = argparse.ArgumentParser(description="红外和热成像融合算法")
    parser.add_argument('--infrared_path', type=str, default='../../test_imgs/ir', help="近红外图像路径")
    parser.add_argument('--thermal_path', type=str, default='../../test_imgs/vi', help="远红外图像路径")
    parser.add_argument('--results_path', type=str, default='./results', help="融合结果保存路径")
    parser.add_argument('--model_path', type=str, default='./models/rknn/fuseint8_v3.rknn', help="模型文件")
    parser.add_argument('--target', type=str, default='rk3588', help="硬件平台")
    parser.add_argument('--rknn_batch', type=int, default=3, help='RKNN batch size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args=args)
