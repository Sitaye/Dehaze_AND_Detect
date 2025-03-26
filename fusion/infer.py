import os
import argparse
import numpy as np
from tqdm import tqdm
from rknn.api import RKNN

from utils import img_read, img_save, BGR2YCrCb, YCrCb2BGR


def main(args):

    # 加载模型
    fusemodel = RKNN()
    fusemodel.load_rknn(args.model_path)
    fusemodel.init_runtime(
        target=args.target,
        core_mask=RKNN.NPU_CORE_0_1_2,
    )

    img_list = os.listdir(args.infrared_path)
    batch_size = args.rknn_batch
    for i in tqdm(range(0, len(img_list), batch_size)):

        # 读取该批次的图像数据
        name_list = img_list[i: min(i + batch_size, len(img_list))]
        Y_list, Cr_list, Cb_list, tm_imgs = [], [], [], []

        # 转化图像数据
        for img_name in name_list:
            ir_img_path = os.path.join(args.infrared_path, img_name)
            tm_img_path = os.path.join(args.thermal_path, img_name)

            ir_img = img_read(path=ir_img_path, is_single=False) # (H, W, C) float32
            tm_img = img_read(path=tm_img_path, is_single=True) # (1, 1, H, W) int8

            Y, Cr, Cb = BGR2YCrCb(ir_img) # Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32

            Y_list.append(Y)
            Cr_list.append(Cr)
            Cb_list.append(Cb)
            tm_imgs.append(tm_img)

        Y_con = np.concatenate(Y_list, axis=0)
        tm_con = np.concatenate(tm_imgs, axis=0)

        # 多批次会出现精度不对齐的问题
        if batch_size > 1:
            Y_con = Y_con.astype(np.float16)
            tm_con = tm_con.astype(np.float16)

        # 模型推理
        inputs = [Y_con, tm_con]
        fuse_imgs = fusemodel.inference(inputs=inputs, data_format='nchw')[0] # int8

        # 保存融合图像
        for fuse_img, Cr, Cb, img_name in zip(fuse_imgs, Cr_list, Cb_list, name_list):
            fuse_img = fuse_img[np.newaxis, ...]
            fuse_img_process = YCrCb2BGR(fuse_img, Cr, Cb) # (H, W, C) float32
            save_path = os.path.join(args.results_path, img_name)
            img_save(fuse_img_process, save_path)

    # 释放资源
    fusemodel.release()


def parse_options():
    parser = argparse.ArgumentParser(description='红外和热成像融合算法')
    parser.add_argument('--model_path', type=str, default='./models/rknn/fuse.rknn', help='模型文件路径')
    parser.add_argument('--target', type=str, default='rk3588', choices=["rk3588", "..."], help='硬件平台')
    parser.add_argument('--rknn_batch', type=int, default=1, help='模型批次')
    parser.add_argument('--infrared_path', type=str, default='./imgs/infrared', help='红外图像路径')
    parser.add_argument('--thermal_path', type=str, default='./imgs/thermal', help='热成像图像路径')
    parser.add_argument('--results_path', type=str, default='./results/fusion', help='融合结果保存路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args=args)
