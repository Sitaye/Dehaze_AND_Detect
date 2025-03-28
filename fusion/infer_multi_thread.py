import os
from concurrent.futures import ThreadPoolExecutor
import queue
from tqdm import tqdm
from rknn.api import RKNN

from utils import load_model, img_read, img_save, BGR2YCrCb, YCrCb2BGR

# 用于线程间传递数据的队列
preprocess_queue = queue.Queue(maxsize=200)
inference_queue = queue.Queue(maxsize=200)

# 进程停止标志
SIGNAL = None

def preprocess_worker(args, img_list, infer_thread_count):
    """
    模型推理进程，对于图像列表中的数据进行转化并放入后续处理队列
    """
    for img_name in img_list:
        ir_img_path = os.path.join(args.infrared_path, img_name)
        tm_img_path = os.path.join(args.thermal_path, img_name)
        
        ir_img = img_read(path=ir_img_path, is_single=False)  # (H, W, C) float32
        tm_img = img_read(path=tm_img_path, is_single=True)   # (1, 1, H, W) float32
        
        Y, Cr, Cb = BGR2YCrCb(ir_img)  # Y (1, 1, H, W) Cr (H, W) Cb (H, W) float32
        
        inputs = [Y, tm_img]
        preprocess_queue.put((inputs, Cr, Cb, img_name))
    
    for _ in range(infer_thread_count):
        preprocess_queue.put(SIGNAL)


def inference_worker(model):
    """
    模型推理进程，对于前序结果队列中的数据进行推理并放入后续处理队列
    """
    while True:
        item = preprocess_queue.get()
        if item == SIGNAL:
            preprocess_queue.task_done()
            inference_queue.put(SIGNAL)
            break

        inputs, Cr, Cb, img_name = item

        try:       
            fuse_img = model.inference(inputs=inputs, data_format='nchw')[0]  # int8
        except Exception:
            preprocess_queue.task_done()
            continue

        inference_queue.put((fuse_img, Cr, Cb, img_name))
        preprocess_queue.task_done()


def postprocess_worker(args, total_img, infer_thread_count):
    """
    结果保存进程，对于前序结果队列中的数据进行转化并保存为图像
    """
    pbar = tqdm(total=total_img)
    finished_inference_workers = 0
    while True:
        item = inference_queue.get()
        if item is SIGNAL:
            finished_inference_workers += 1
            inference_queue.task_done()
            if finished_inference_workers == infer_thread_count:
                break
            continue

        fuse_img, Cr, Cb, img_name = item
        fuse_img_process = YCrCb2BGR(fuse_img, Cr, Cb)  # (H, W, C) float32
        
        save_path = os.path.join(args.results_path, img_name)
        img_save(fuse_img_process, save_path)
        
        pbar.update(1)
        inference_queue.task_done()
    
    pbar.close()


def main(args):

    os.makedirs(args.results_path, exist_ok=True)

    img_list = os.listdir(args.infrared_path)

    # 创建线程池
    with ThreadPoolExecutor(max_workers=5) as pool:
        infer_thread_count = 3

        # 图像处理
        pool.submit(preprocess_worker, args, img_list, infer_thread_count)

        # 模型推理
        core_masks = [RKNN.NPU_CORE_2, RKNN.NPU_CORE_1, RKNN.NPU_CORE_0]
        models = []
        for count in range(infer_thread_count):
            fusemodel = load_model(
                model_path=args.model_path,
                target=args.target,
                core_mask=core_masks[count],
            )
            models.append(fusemodel)
            pool.submit(inference_worker, fusemodel)

        # 结果保存
        pool.submit(postprocess_worker, args, len(img_list), infer_thread_count)

        # 等待任务结束
        preprocess_queue.join()
        inference_queue.join()

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
