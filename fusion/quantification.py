from rknn.api import RKNN
import numpy
import os
from utils import *
from tqdm import tqdm
import argparse
from typing import List
H = 540
W = 960

def main(args):

    algorithm1 = args.algorithm
    method1 = args.method
    dtype1 = args.dtype
    fuse_model_path = args.fuse_model_path
    fuse_input_list = args.fuse_input_list
    is_do_qua1 = args.qua
    datasets1 = args.datasets_file
    export_model_name = args.export_model_name

    fusemodel = RKNN(verbose=False)
    fusemodel.config(
        target_platform='rk3588',
        quant_img_RGB2BGR=False,
        quantized_algorithm=algorithm1,
        quantized_method=method1,
        quantized_dtype=dtype1
    )
    
    fusemodel.load_pytorch(fuse_model_path, fuse_input_list)
    if is_do_qua1:
        fusemodel.build(do_quantization=True, dataset=datasets1)
    else:
        fusemodel.build(do_quantization=False)

    fusemodel.export_rknn(export_model_name)
    fusemodel.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='w8a8', help='qunatized_dtype 量化类型')
    parser.add_argument('--method', type=str, default='channel', help='qunatized method 量化方法 layer or channel')
    parser.add_argument('--algorithm', type=str, default='normal', help='quantized algorithm 量化算法 normal or mmse or kl')
    parser.add_argument('--fuse_model_path', type=str, default='./fusescript.pt', help='待量化模型的路径')
    parser.add_argument('--fuse_input_list',type=List[List[int]], default=[[1, 1, H, W], [1, 1, H, W]], help='输入数据格式')
    parser.add_argument('--qua', type=bool, default=True, help='是否做量化')
    parser.add_argument('--datasets_file', type=str, default='./datasets.txt', help='量化数据集组织文件')
    parser.add_argument('--export_model_name', type=str, default='./fuseint8_v4.rknn', help='导出模型路径')
    args = parser.parse_args()

    main(args)

