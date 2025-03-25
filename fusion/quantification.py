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
    rknn_batch_size = args.rknn_batch

    # for sublist in fuse_input_list:
    #     sublist[0] = rknn_batch_size
    # print(fuse_input_list)

    fusemodel = RKNN(verbose=False)
    fusemodel.config(
        target_platform='rk3588',
        quant_img_RGB2BGR=False,
        quantized_algorithm=algorithm1,
        quantized_method=method1,
        quantized_dtype=dtype1
    )
    
    ret = fusemodel.load_pytorch(fuse_model_path, fuse_input_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    else:
        print('Load successfully')
    
    if is_do_qua1:
        ret = fusemodel.build(do_quantization=True, dataset=datasets1, rknn_batch_size=rknn_batch_size)
    else:
        ret = fusemodel.build(do_quantization=False, rknn_batch_size=rknn_batch_size)

    if ret != 0:
        print('Build model failed!')
        exit(ret)
    else:
        print('Build successfully')

    fusemodel.export_rknn(export_model_name)
    fusemodel.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='w8a8', help='qunatized_dtype 量化类型')
    parser.add_argument('--method', type=str, default='channel', help='qunatized method 量化方法 layer or channel')
    parser.add_argument('--algorithm', type=str, default='normal', help='quantized algorithm 量化算法 normal or mmse or kl')
    parser.add_argument('--fuse_model_path', type=str, default='./fusescript.pt', help='待量化模型的路径')
    parser.add_argument('--fuse_input_list',type=List[List[int]], default=[[1, 1, H, W], [1, 1, H, W]], help='输入数据格式')
    parser.add_argument('--qua', type=bool, default=False, help='是否做量化')
    parser.add_argument('--datasets_file', type=str, default='./datasets.txt', help='量化数据集组织文件')
    parser.add_argument('--export_model_name', type=str, default='./fuseint8_batch3.rknn', help='导出模型路径')
    parser.add_argument('--rknn_batch', type=int, default=3, help='rknn batch size')
    args = parser.parse_args()

    main(args)

