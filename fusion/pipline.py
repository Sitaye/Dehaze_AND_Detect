import os
import cv2
import numpy as np
from tqdm import tqdm
from queue import Empty, Queue
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor

from fusion.utils import BGR2YCrCb, YCrCb2RGB, draw, img_convert, img_read, load_model


class Pipeline:
    def __init__(self, config):
        # 常量参数
        self.H = config["output"]["H"]
        self.W = config["output"]["W"]
        self.IMG_SIZE = config["output"]["IMG_SIZE"]
        self.FPS = config["output"]["FPS"]
        self.SIGNAL = object()
        self.FINAL = float("inf")

        # 数据参数
        self.infrared_image_path = config["output"]["path"]["image"]["infrared"]
        self.thermal_image_path = config["output"]["path"]["image"]["thermal"]
        self.infrared_video_path = config["output"]["path"]["video"]["infrared"]
        self.thermal_video_path = config["output"]["path"]["video"]["thermal"]
        self.results_path = config["output"]["path"]["results"]
        
        # RKNN 参数
        self.target = config["rknn"]["target"]
        self.fuse_model_path = config["rknn"]["model"]["fuse"]["path"]
        self.yolo_model_path = config["rknn"]["model"]["yolo"]["path"]

        # 线程参数
        self.max_workers = config["thread"]["max_workers"]
        self.conf_threshold = config["rknn"]["model"]["yolo"]["conf_threshold"]
        self.infer_thread_count = config["thread"]["infer"]["max_workers"]
        self.detect_thread_count = config["thread"]["detect"]["max_workers"]

    def preprocess(self, mode):
        if mode == 'frame':
            img_list = os.listdir(self.infrared_image_path)
            self.pbar = tqdm(total=len(img_list))

            for idx, img_name in enumerate(img_list):
                ir_img_path = os.path.join(self.infrared_image_path, img_name)
                tm_img_path = os.path.join(self.thermal_image_path, img_name)

                ir_img = img_read(path=ir_img_path, W=self.W, H=self.H, is_single=False) # (H, W, 3) float32
                tm_img = img_read(path=tm_img_path, W=self.W, H=self.H, is_single=True) # (1, 1, H, W) float32
                Y, Cr, Cb = BGR2YCrCb(ir_img) # (1, 1, H, W), (H, W), (H, W) float32
                
                inputs = [Y, tm_img] # [(1, 1, H, W), (1, 1, H, W)] [float32, float32]
                self.preprocess_queue.put((idx, (inputs, Cr, Cb, img_name)))

        elif mode == 'video':
            cap_ir = cv2.VideoCapture(self.infrared_video_path)
            cap_tm = cv2.VideoCapture(self.thermal_video_path)
            frame_count = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
            self.pbar = tqdm(total=frame_count)

            idx = 0
            while cap_ir.isOpened() and cap_tm.isOpened():
                ret_ir, ir_img = cap_ir.read()
                ret_tm, tm_img = cap_tm.read()
                if not ret_ir or not ret_tm:
                    break

                ir_img = img_convert(img=ir_img, W=self.W, H=self.H, is_single=False) # (H, W, 3) float32
                tm_img = img_convert(img=tm_img, W=self.W, H=self.H, is_single=True) # (1, 1, H, W) float32
                Y, Cr, Cb = BGR2YCrCb(ir_img) # (1, 1, H, W), (H, W), (H, W) float32

                inputs = [Y, tm_img] # [(1, 1, H, W), (1, 1, H, W)] [float32, float32]
                self.preprocess_queue.put((idx, (inputs, Cr, Cb, None)))
                idx += 1

            cap_ir.release()
            cap_tm.release()

        # 通知融合线程数据传输完毕
        for _ in range(self.infer_thread_count):
            self.preprocess_queue.put((self.FINAL, self.SIGNAL))

    def inference(self, fusemodel):
        while True:
            idx, item = self.preprocess_queue.get()
            if item is self.SIGNAL:
                # 得到数据传输完毕的通知，关闭线程，并通知检测线程有一个融合线程已经处理完毕
                self.inference_queue.put((self.FINAL, self.SIGNAL))
                self.preprocess_queue.task_done()
                break

            inputs, Cr, Cb, img_name = item # [(1, 1, H, W), (1, 1, H, W)] [float32, float32]
            fuse_img = fusemodel.inference(inputs=inputs, data_format="nchw")[0] # (1, 1, H, W) float32

            self.inference_queue.put((idx, (fuse_img, Cr, Cb, img_name)))
            self.preprocess_queue.task_done()

    def detection(self, yolomodel):
        while True:
            # 如果此时融合线程已经处理完毕，则通知后处理线程有一个检测线程已经处理完毕
            if self.infer_finished_event.is_set():
                self.detection_queue.put((self.FINAL, self.SIGNAL))
                break
            try:
                idx, item = self.inference_queue.get(timeout=0.5)
                if item is self.SIGNAL:
                    with self.infer_finished_lock:  # 临界资源互斥锁
                        self.infer_finished_count += 1
                        self.inference_queue.task_done()
                        # 所有的融合线程已经处理完毕，则标记
                        if self.infer_finished_count == self.infer_thread_count:
                            self.infer_finished_event.set()
                            self.detection_queue.put((self.FINAL, self.SIGNAL))
                            break
                        continue

                fuse_img, Cr, Cb, img_name = item # (1, 1, H, W), (H, W), (H, W) float32

                fuse_img = YCrCb2RGB(Y=fuse_img, Cr=Cr, Cb=Cb)  # (H, W, 3) float32
                fuse_img = np.clip(fuse_img * 255.0, 0, 255).astype(np.uint8)  # (H, W, 3) uint8
                fuse_img = cv2.resize(fuse_img, (self.IMG_SIZE, self.IMG_SIZE))  # (IMG_SIZE, IMG_SIZE, 3) uint8
                fuse_img = np.transpose(fuse_img, (2, 0, 1)) # (3, IMG_SIZE, IMG_SIZE) uint8

                fuse_img_process = np.expand_dims(fuse_img, 0) # (1, 3, IMG_SIZE, IMG_SIZE) uint8

                output = yolomodel.inference(inputs=[fuse_img_process], data_format=["nchw"]) # (1, 1, 5, 8400) uint8

                self.detection_queue.put((idx, (output[0][0], fuse_img, img_name)))
                self.inference_queue.task_done()
            except Empty:
                continue
            
    def postprocess(self, mode):
        expected_index = 0
        buffer = {}
        if mode == 'video':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(os.path.join(self.results_path, "results.mp4"), fourcc, self.FPS, (self.W, self.H))
        while True:
            # 检测线程处理完毕则直接退出
            if self.detect_finished_event.is_set():
                break
            try:
                idx, item = self.detection_queue.get(timeout=0.5)
                if item is self.SIGNAL:
                    with self.detect_finished_lock:  # 临界资源互斥锁
                        self.detect_finished_count += 1
                        self.detection_queue.task_done()
                        # 所有的检测线程已经处理完毕，则标记
                        if self.detect_finished_count == self.detect_thread_count:
                            self.detect_finished_event.set()
                            if mode == 'video':
                                video_writer.release()
                            break
                        continue

                # 将图像放入缓冲池
                buffer[idx] = item
                while expected_index in buffer:
                    output, fuse_img, img_name = buffer.pop(expected_index) # (5, 8400) uint8
                    fuse_img = draw(
                        fuse_img=fuse_img,
                        W=self.W,
                        H=self.H, 
                        IMG_SIZE=self.IMG_SIZE,
                        output=output,
                        conf_threshold=self.conf_threshold,
                    )
                    if mode == 'frame':
                        save_path = os.path.join(self.results_path, img_name)
                        cv2.imwrite(save_path, fuse_img)
                    elif mode == 'video':
                        video_writer.write(fuse_img)

                    expected_index += 1
                    self.pbar.update(1)
                    self.detection_queue.task_done()

            except Empty:
                continue

    def reset(self):
        self.models = []
        self.infer_finished_count = 0
        self.detect_finished_count = 0
        self.infer_finished_lock = Lock()
        self.detect_finished_lock = Lock()
        self.infer_finished_event = Event()
        self.detect_finished_event = Event()
        self.preprocess_queue = Queue(maxsize=300)
        self.inference_queue = Queue()
        self.detection_queue = Queue()

    def release(self):
        # 释放 RKNN 模型
        for model in self.models:
            model.release()

        # 释放 pbar
        if self.pbar is not None:
            self.pbar.close()

    def run(self, mode):
        # 重置参数
        self.reset()

        # 创建处理结果
        if mode == 'frame' or mode == 'video':
            os.makedirs(self.results_path, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 图像预处理
            pool.submit(self.preprocess, mode)

            # 图像融合
            for _ in range(self.infer_thread_count):
                fusemodel = load_model(
                    model_path=self.fuse_model_path,
                    target=self.target,
                )
                self.models.append(fusemodel)
                pool.submit(self.inference, fusemodel)

            # 人体检测
            for _ in range(self.detect_thread_count):
                yolomodel = load_model(
                    model_path=self.yolo_model_path,
                    target=self.target,
                )
                self.models.append(yolomodel)
                pool.submit(self.detection, yolomodel)

            # 图像输出
            pool.submit(self.postprocess, mode)

            # 等待任务完成
            self.preprocess_queue.join()
            self.inference_queue.join()
            self.detection_queue.join()

            # 释放资源
            self.release()