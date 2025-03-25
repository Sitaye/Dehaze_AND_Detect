from rknn.api import RKNN
import numpy
import os
from utils import *
from tqdm import tqdm
import argparse


def main(args):

    H = 540
    W = 960
    target1 = None

    vi_path = args.vi_path
    ir_path = args.ir_path
    res_path = args.res_path

    process_type = 'img'
    export_model_name = args.model_path

    fusemodel = RKNN(verbose=False)
    fusemodel.config(target_platform='rk3588',quant_img_RGB2BGR=False,)
    fusemodel.export_rknn(export_model_name)

    

    fusemodel.init_runtime(target=target1)

    if process_type == 'img':
        img_list = os.listdir(vi_path)

        for img_name in tqdm(img_list):
            vi_img_path = os.path.join(vi_path, img_name)
            ir_img_path = os.path.join(ir_path, img_name)

            vi_img = img_read(vi_img_path, is_vis=True) # (H, W, C) float32
            ir_img = img_read(ir_img_path, is_vis=False) # (1, 1, H, W) int8

            Y, Cr, Cb = BGR2YCrCb(vi_img)# Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32
            inputs = [Y, ir_img]
            fuse_img = fusemodel.inference(inputs=inputs, data_format='nchw')[0] # int8
            fuse_img = YCrCb2BGR(fuse_img, Cr, Cb) # (H, W, C) float32
            save_path = os.path.join(res_path, img_name)
            img_save(fuse_img, save_path)

    fusemodel.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vi_path', type=str, default='./test_imgs/vi', help="可见光图像")
    parser.add_argument('--ir_path', type=str, default='./test_imgs/ir', help="红外图像")
    parser.add_argument('--res_path', type=str, default='./test_imgs/fuse', help='保存结果')
    parser.add_argument('--model_path', type=str, default='./fuseint8_v3.rknn', help='模型文件')
    args = parser.parse_args()

    main(args=args)


    
