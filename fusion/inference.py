import os
import argparse
from tqdm import tqdm
from rknn.api import RKNN

from utils import img_read, BGR2YCrCb, YCrCb2BGR, img_save


def main(args):

    # 加载模型
    fusemodel = RKNN(verbose=False)
    fusemodel.load_rknn(args.model_path)
    fusemodel.init_runtime(
        target=args.target,
        core_mask=RKNN.NPU_CORE_ALL,
    )

    # 读取图像数据
    img_list = os.listdir(args.infrared_path)

    # 遍历融合图像数据
    for img_name in tqdm(img_list):
        ir_img_path = os.path.join(args.infrared_path, img_name)
        tm_img_path = os.path.join(args.thermal_path, img_name)

        ir_img = img_read(path=ir_img_path, is_vis=True) # (H, W, C) float32
        tm_img = img_read(path=tm_img_path, is_vis=False) # (1, 1, H, W) int8

        Y, Cr, Cb = BGR2YCrCb(ir_img) # Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32
        inputs = [Y, tm_img]
        fuse_img = fusemodel.inference(inputs=inputs, data_format='nchw')[0] # int8

        fuse_img_process = YCrCb2BGR(fuse_img, Cr, Cb) # (H, W, C) float32
        save_path = os.path.join(args.results_path, img_name)
        img_save(fuse_img_process, save_path)

    # 释放资源
    fusemodel.release()


def parse_options():
    parser = argparse.ArgumentParser(description="红外和热成像融合算法")
    parser.add_argument('--infrared_path', type=str, default='./data/infrared', help="近红外图像路径")
    parser.add_argument('--thermal_path', type=str, default='./data/thermal', help="远红外图像路径")
    parser.add_argument('--results_path', type=str, default='./results', help="融合结果保存路径")
    parser.add_argument('--model_path', type=str, default='./models/rknn/fuseint8_v3.rknn', help="模型文件")
    parser.add_argument('--target', type=str, default='rk3588', help="硬件平台")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args=args)
