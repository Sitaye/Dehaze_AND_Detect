import argparse
from typing import List
from rknn.api import RKNN

# 图像大小
H = 540
W = 960

def main(args):

    # 定义模型
    fusemodel = RKNN(verbose=False)
    fusemodel.config(
        quantized_dtype=args.dtype,
        quantized_algorithm=args.algorithm,
        quantized_method=args.method,
        target_platform=args.target,
    )
    
    # 加载模型
    ret = fusemodel.load_pytorch(
        args.model_path,
        args.input_format,
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    else:
        print('Load successfully')
    
    # 量化模型
    ret = fusemodel.build(
        do_quantization=args.do_quatize,
        dataset=args.dataset_file,
        rknn_batch_size=args.batch_size,
    )
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    else:
        print('Build successfully')

    # 导出处理后的模型
    ret = fusemodel.export_rknn(args.export_model_name)
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    else:
        print('Export successfully')

    # 释放资源
    fusemodel.release()


def parse_options():
    parser = argparse.ArgumentParser(dedescription='模型转化和量化')
    parser.add_argument('--dtype', type=str, default='w8a8', help='量化精度 "w8a8", "w8a16", "w16a16i", "w16a16i_dfp"')
    parser.add_argument('--algorithm', type=str, default='normal', help='量化算法 "normal", "mmse", "kl_divergence"')
    parser.add_argument('--method', type=str, default='channel', help='量化方法 "channel", "layer"')
    parser.add_argument('--target', type=str, default='rk3588', help='硬件平台 "rk3588", ...')
    parser.add_argument('--model_path', type=str, default='./fusescript.pt', help='待量化的模型路径')
    parser.add_argument('--input_format',type=List[List[int]], default=[[1, 1, H, W], [1, 1, H, W]], help='输入数据格式 "[[1, 1, H, W], [1, 1, H, W]]"')
    parser.add_argument('--do_quatize', type=bool, default=False, help='是否做量化 "False", "True"')
    parser.add_argument('--dataset_file', type=str, default='./dataset.txt', help='量化数据集组织文件')
    parser.add_argument('--batch_size', type=int, default=None, help='多批次量化')
    parser.add_argument('--export_model_name', type=str, default='./models/fuse.rknn', help='导出模型路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args)

