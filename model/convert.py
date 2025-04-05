import argparse
from typing import List
from rknn.api import RKNN

# 图像大小
H = 480
W = 854

def main(args):

    # 定义模型
    fusemodel = RKNN()
    
    # 加载模型
    if args.model_path.endswith('.onnx'):
        fusemodel.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            quantized_dtype=args.dtype,
            quantized_algorithm=args.algorithm,
            quantized_method=args.method,
            target_platform=args.target,
            model_pruning=args.do_pruning,
            single_core_mode=args.do_single_core_mode,
        )
        ret = fusemodel.load_onnx(model=args.model_path)
    elif args.model_path.endswith('.pt'):
        fusemodel.config(
            quantized_dtype=args.dtype,
            quantized_algorithm=args.algorithm,
            quantized_method=args.method,
            target_platform=args.target,
            model_pruning=args.do_pruning,
            single_core_mode=args.do_single_core_mode,
        )
        ret = fusemodel.load_pytorch(
            model=args.model_path,
            input_size_list=args.input_format,
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
    parser = argparse.ArgumentParser(description='模型转化和量化')
    parser.add_argument('--dtype', type=str, default='w8a8', choices=['w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp'], help='量化精度')
    parser.add_argument('--algorithm', type=str, default='kl_divergence', choices=['normal', 'mmse', 'kl_divergence'], help='量化算法')
    parser.add_argument('--method', type=str, default='channel', choices=['channel', 'layer'], help='量化方法')
    parser.add_argument('--target', type=str, default='rk3588', choices=['rk3588', '...'], help='硬件平台')
    parser.add_argument('--do_pruning', type=bool, default=True, choices=['True', 'False'], help='是否剪枝')
    parser.add_argument('--model_path', type=str, default='model/origin/fusescript.pt', help='源模型路径')
    parser.add_argument('--input_format',type=List[List[int]], default=[[1, 1, H, W], [1, 1, H, W]], help='数据输入格式')
    parser.add_argument('--do_quatize', type=bool, default=True, choices=['True', 'False'], help='是否量化')
    parser.add_argument('--do_single_core_mode', type=bool, default=True, choices=['True', 'False'], help='是否开启单核模式')
    parser.add_argument('--dataset_file', type=str, default='model/dataset.txt', help='量化数据集组织文件')
    parser.add_argument('--export_model_name', type=str, default='model/rknn/fuse_int8_480x854_kl.rknn', help='模型导出路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args)

