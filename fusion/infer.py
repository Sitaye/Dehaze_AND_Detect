import os
from tqdm import tqdm
from rknn.api import RKNN

from utils import load_model, img_read, img_save, BGR2YCrCb, YCrCb2BGR

def main(args):

    # 加载模型
    fusemodel = load_model(
        model_path=args.model_path,
        target=args.target,
        core_mask=RKNN.NPU_CORE_0_1_2,
    )

    img_list = os.listdir(args.infrared_path)

    for img_name in tqdm(img_list):
        # 转化图像数据
        ir_img_path = os.path.join(args.infrared_path, img_name)
        tm_img_path = os.path.join(args.thermal_path, img_name)

        ir_img = img_read(path=ir_img_path, is_single=False) # (H, W, C)s
        tm_img = img_read(path=tm_img_path, is_single=True) # (1, 1, H, W)s

        Y, Cr, Cb = BGR2YCrCb(ir_img) # Y (1, 1, H, W), Cr (H, W), Cb (H, W)s

        # 模型推理
        inputs = [Y, tm_img]
        fuse_img = fusemodel.inference(inputs=inputs, data_format='nchw')[0]

        # 保存融合图像
        fuse_img_process = YCrCb2BGR(fuse_img, Cr, Cb) # (H, W, C)s
        save_path = os.path.join(args.results_path, img_name)
        img_save(fuse_img_process, save_path)

    # 释放资源
    fusemodel.release()


def parse_options():
    import argparse
    parser = argparse.ArgumentParser(description='红外和热成像融合算法')
    parser.add_argument('--model_path', type=str, default='./models/rknn/fuse_int8_resize.rknn', help='模型文件路径')
    parser.add_argument('--target', type=str, default='rk3588', choices=["rk3588", "..."], help='硬件平台')
    parser.add_argument('-infrared_path', type=str, default='./imgs/infrared/', help='红外图像路径')
    parser.add_argument('-thermal_path', type=str, default='./imgs/thermal/', help='热成像图像路径')
    parser.add_argument('-results_path', type=str, default='./results', help='识别结果保存路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args=args)
