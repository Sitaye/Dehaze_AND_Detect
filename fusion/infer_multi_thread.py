import os
import cv2
import queue
import numpy as np
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from utils import YCrCb2RGB, draw, load_model, img_read, BGR2YCrCb, yolov5_post_process


class FusionPipeline:
    def __init__(self, args):
        self.args = args
        self.H = 480
        self.W = 854
        self.IMG_SIZE = 640
        self.SIGNAL = object()

        self.img_list = os.listdir(self.args.infrared_path)
        self.infer_finished_count = 0
        self.infer_finished_lock = Lock()
        self.detect_finished_count = 0
        self.detect_finished_lock = Lock()

        self.max_workers = 15
        self.models = []
        self.infer_thread_count = 3
        self.detect_thread_count = 6

        self.pbar = tqdm(total=len(self.img_list))
        self.preprocess_queue = queue.Queue(maxsize=500)
        self.inference_queue = queue.Queue()
        self.detection_queue = queue.Queue()

    def preprocess_worker(self):
        for img_name in self.img_list:
            ir_img_path = os.path.join(self.args.infrared_path, img_name)
            tm_img_path = os.path.join(self.args.thermal_path, img_name)

            ir_img = img_read(path=ir_img_path, is_single=False)
            tm_img = img_read(path=tm_img_path, is_single=True)

            Y, Cr, Cb = BGR2YCrCb(ir_img)
            inputs = [Y, tm_img]
            self.preprocess_queue.put((inputs, Cr, Cb, img_name))

        # 通知融合线程数据传输完毕
        for _ in range(self.infer_thread_count):
            self.preprocess_queue.put(self.SIGNAL)

    def inference_worker(self, fusemodel):
        while True:
            item = self.preprocess_queue.get()
            if item is self.SIGNAL:
                # 得到数据传输完毕的通知，关闭线程，并通知检测进程有一个融合进程已经融合完毕
                self.inference_queue.put(self.SIGNAL)
                self.preprocess_queue.task_done()
                break

            inputs, Cr, Cb, img_name = item
            fuse_img = fusemodel.inference(inputs=inputs, data_format="nchw")[0]

            self.inference_queue.put((fuse_img, Cr, Cb, img_name))
            self.preprocess_queue.task_done()

    def detection_worker(self, yolomodel):
        while True:
            item = self.inference_queue.get()
            if item is self.SIGNAL:
                with self.infer_finished_lock:  # 临界资源互斥锁
                    # 当前为融合进程的最后一个信号量
                    if self.infer_finished_count == self.infer_thread_count - 1:
                        # 通知其他检测线程可以关闭
                        for _ in range(self.detect_thread_count - 1):
                            self.inference_queue.put(self.SIGNAL)
                        # 通知后处理线程有一个检测进程已经关闭
                        self.detection_queue.put(self.SIGNAL)
                        self.infer_finished_count += 1
                        self.inference_queue.task_done()
                        break
                    # 当前为同级进程传来的通知信号量
                    if self.infer_finished_count == self.infer_thread_count:
                        # 通知后处理线程有一个检测进程已经关闭
                        self.detection_queue.put(self.SIGNAL)
                        self.inference_queue.task_done()
                        break
                    # 记录融合进程有一个已经关闭
                    self.infer_finished_count += 1
                    self.inference_queue.task_done()
                    continue

            fuse_img, Cr, Cb, img_name = item

            fuse_img = YCrCb2RGB(fuse_img, Cr, Cb)  # (H, W, 3) float32
            fuse_img = np.clip(fuse_img * 255.0, 0, 255).astype(
                np.uint8
            )  # (H, W, 3) uint8
            fuse_img = cv2.resize(
                fuse_img, (self.IMG_SIZE, self.IMG_SIZE)
            )  # (IMG_SIZE, IMG_SIZE, 3) uint8

            fuse_img_process = np.expand_dims(fuse_img, 0)

            outputs = yolomodel.inference(
                inputs=[fuse_img_process], data_format=["nhwc"]
            )

            self.detection_queue.put((outputs, fuse_img, img_name))
            self.inference_queue.task_done()

    def postprocess_worker(self):
        while True:
            item = self.detection_queue.get()
            if item is self.SIGNAL:
                with self.detect_finished_lock:  # 临界资源互斥锁
                    # 当前为检测进程的最后一个信号量
                    if self.detect_finished_count == self.detect_thread_count - 1:
                        self.detection_queue.task_done()
                        break
                    # 记录检测进程有一个已经关闭
                    self.detect_finished_count += 1
                    self.detection_queue.task_done()
                    continue

            outputs, fuse_img, img_name = item

            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]

            input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
            input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
            input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

            input_data = []
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)

            fuse_img = cv2.cvtColor(fuse_img, cv2.COLOR_RGB2BGR)

            if boxes is not None:
                draw(fuse_img, boxes, scores, classes)

            fuse_img = cv2.resize(fuse_img, (self.W, self.H))
            save_path = os.path.join(self.args.results_path, img_name)
            cv2.imwrite(save_path, fuse_img)

            self.pbar.update(1)
            self.detection_queue.task_done()

    def release(self):
        # 释放 RKNN 模型
        for model in self.models:
            model.release()
        self.models = []

        # 释放 pbar
        if self.pbar is not None:
            self.pbar.close()

    def run(self):
        os.makedirs(args.results_path, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 图像预处理
            pool.submit(self.preprocess_worker)

            # 图像融合
            for _ in range(self.infer_thread_count):
                fusemodel = load_model(
                    model_path=self.args.fuse_model_path,
                    target=self.args.target,
                )
                self.models.append(fusemodel)
                pool.submit(self.inference_worker, fusemodel)

            # 人体检测
            for _ in range(self.detect_thread_count):
                yolomodel = load_model(
                    model_path=self.args.yolo_model_path,
                    target=self.args.target,
                )
                self.models.append(yolomodel)
                pool.submit(self.detection_worker, yolomodel)

            # 图像保存
            pool.submit(self.postprocess_worker)

            # 等待任务完成
            self.preprocess_queue.join()
            self.inference_queue.join()
            self.detection_queue.join()

            # 释放资源
            self.release()


def parse_options():
    import argparse

    parser = argparse.ArgumentParser(description="红外和热成像融合算法")
    parser.add_argument("--rknn_model_path", type=str, default="./models/rknn/fuse.rknn", help="Fuse 模型文件路径",)
    parser.add_argument("--yolo_model_path", type=str, default="./models/rknn/yolo11.rknn", help="Yolo 模型文件路径",)
    parser.add_argument("--target", type=str, default="rk3588", choices=["rk3588", "..."], help="硬件平台",)
    parser.add_argument("--infrared_path", type=str, default="./imgs/infrared", help="红外图像路径",)
    parser.add_argument("--thermal_path", type=str, default="./imgs/thermal", help="热成像图像路径",)
    parser.add_argument("--results_path", type=str, default="./results/fusion", help="识别结果保存路径",)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_options()
    pipeline = FusionPipeline(args)
    pipeline.run()
