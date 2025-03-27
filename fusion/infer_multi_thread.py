import os
import threading
from threading import Thread
import queue
from tqdm import tqdm
from rknn.api import RKNN

from utils import load_model, img_read, img_save, BGR2YCrCb, YCrCb2BGR

# 用于线程间传递数据的队列
preprocess_queue = queue.Queue()
inference_queue = queue.Queue()

# 线程停止标志
stop_event = threading.Event()


def preprocess_worker(args):
    """
    模型推理进程，对于图像列表中的数据进行转化并放入后续处理队列
    """
    img_list = os.listdir(args.infrared_path)
    
    for img_name in img_list:
        ir_img_path = os.path.join(args.infrared_path, img_name)
        tm_img_path = os.path.join(args.thermal_path, img_name)
        
        ir_img = img_read(path=ir_img_path, is_single=False)  # (H, W, C) float32
        tm_img = img_read(path=tm_img_path, is_single=True)   # (1, 1, H, W) float32
        
        Y, Cr, Cb = BGR2YCrCb(ir_img)  # Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32
        inputs = [Y, tm_img]
        
        preprocess_queue.put((inputs, Cr, Cb, img_name))
    
    stop_event.set()


def inference_worker(model):
    """
    模型推理进程，对于前序结果队列中的数据进行推理并放入后续处理队列
    """
    while not stop_event.is_set() or preprocess_queue.empty():
        try:
            inputs, Cr, Cb, img_name = preprocess_queue.get(timeout=1)            
            
            fuse_img = model.inference(inputs=inputs, data_format='nchw')[0]  # int8
            
            inference_queue.put((fuse_img, Cr, Cb, img_name))
            preprocess_queue.task_done()

        except queue.Empty:
            continue


def postprocess_worker(args):
    """
    结果保存进程，对于前序结果队列中的数据进行转化并保存为图像
    """
    pbar = tqdm(total=len(os.listdir(args.infrared_path)))
    
    while not stop_event.is_set() or inference_queue.empty():
        try:
            fuse_img, Cr, Cb, img_name = inference_queue.get(timeout=1)
            fuse_img_process = YCrCb2BGR(fuse_img, Cr, Cb)  # (H, W, C) float32
            
            save_path = os.path.join(args.results_path, img_name)
            img_save(fuse_img_process, save_path)
            
            pbar.update(1)
            inference_queue.task_done()

        except queue.Empty:
            continue
    
    pbar.close()


def main(args):

    # 确保结果目录存在
    os.makedirs(args.results_path, exist_ok=True)

    # 创建线程和模型
    threads = []
    models = []

    # 图像处理线程
    preprocess_thread = Thread(
        target=preprocess_worker,
        args=(args,),
        daemon=True,
    )
    threads.append(preprocess_thread)
    
    # 模型推理线程
    infer_thread_count = 3
    for count in range(infer_thread_count):
        core_masks = [RKNN.NPU_CORE_2, RKNN.NPU_CORE_1, RKNN.NPU_CORE_0]
        fusemodel = load_model(
            model_path=args.model_path,
            target=args.target,
            core_mask=core_masks[count],
        )
        models.append(fusemodel)
        infer_thread = Thread(
            target=inference_worker,
            args=(fusemodel,),
            daemon=True,
        )
        threads.append(infer_thread)
    
    # 结果保存线程
    postprocess_thread = Thread(
        target=postprocess_worker,
        args=(args,),
        daemon=True,
    )
    threads.append(postprocess_thread)
    
    # 维护线程
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # 释放资源
    for model in models:
        model.release()


def parse_options():
    import argparse
    parser = argparse.ArgumentParser(description='红外和热成像融合算法')
    parser.add_argument('--model_path', type=str, default='./models/rknn/fuse.rknn', help='模型文件路径')
    parser.add_argument('--target', type=str, default='rk3588', choices=["rk3588", "..."], help='硬件平台')
    parser.add_argument('--infrared_path', type=str, default='./imgs/infrared', help='红外图像路径')
    parser.add_argument('--thermal_path', type=str, default='./imgs/thermal', help='热成像图像路径')
    parser.add_argument('--results_path', type=str, default='./results/fusion', help='融合结果保存路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_options()
    main(args=args)
