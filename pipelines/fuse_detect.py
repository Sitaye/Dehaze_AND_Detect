import os
import re
import cv2
import numpy as np
from queue import Empty, Queue
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor

from utils import BGR2YCrCb, YCrCb2RGB, draw, img_convert, load_model


class FD:
    def __init__(self, config, stop_event=Event()):
        # 常量参数
        self.H = config["output"]["H"]
        self.W = config["output"]["W"]
        self.IMG_SIZE = config["output"]["IMG_SIZE"]
        self.SIGNAL = object()
        self.FINAL = float("inf")

        # 数据参数
        self.infrared_video_path = config["output"]["path"]["video"]["infrared"]
        self.thermal_video_path = config["output"]["path"]["video"]["thermal"]
        
        # RKNN 参数
        self.target = config["rknn"]["target"]
        self.fuse_model_path = config["rknn"]["model"]["fuse"]["path"]
        self.yolo_model_path = config["rknn"]["model"]["yolo"]["path"]

        # 线程参数
        self.stop_event = stop_event
        self.maxsize = config["thread"]["maxsize"]
        self.max_workers = config["thread"]["max_workers"]
        self.conf_threshold = config["rknn"]["model"]["yolo"]["conf_threshold"]
        self.fuse_thread_count = config["thread"]["fuse"]["max_workers"] + 1
        self.detect_thread_count = config["thread"]["detect"]["max_workers"] + 4

    def preprocess(self):
        # 获取视频流
        cap_ir = cv2.VideoCapture(self.infrared_video_path)
        cap_tm = cv2.VideoCapture(self.thermal_video_path)

        # 获取帧率
        ir_fps = cap_ir.get(cv2.CAP_PROP_FPS)
        tm_fps = cap_tm.get(cv2.CAP_PROP_FPS)

        target_fps = 25
        frame_interval = 1.0 / target_fps

        ir_next_time = 0.0
        tm_next_time = 0.0

        ir_frame_id = 0
        tm_frame_id = 0

        # 空间对齐参数
        infrared_file_name = os.path.basename(self.infrared_video_path)
        pattern = r"^(smoked1|smoked2|smoked3)_"
        infrared_match = re.match(pattern, infrared_file_name)

        if infrared_match:
            if infrared_match.group(1) == "smoked1":
                ir_re = [22, 662]
                tm_re = [76, 495]
            elif infrared_match.group(1) == "smoked2":
                ir_re = [18, 658]
                tm_re = [69, 488]
            elif infrared_match.group(1) == "smoked3":
                ir_re = [18, 658]
                tm_re = [71, 490]
        else:
            ir_re = None
            tm_re = None
            
        idx = 0
        while cap_ir.isOpened() and cap_tm.isOpened():
            # 外部中断
            if self.stop_event.is_set():
                break

            # 获取对齐的时间
            ir_time = ir_frame_id / ir_fps
            tm_time = tm_frame_id / tm_fps
            
            # 进行时间戳对齐
            if ir_time >= ir_next_time and tm_time >= tm_next_time:
                # 读取视频帧
                ret_ir, ir_img = cap_ir.read()
                ret_tm, tm_img = cap_tm.read()
                if not ret_ir or not ret_tm:
                    break
                
                # 应用空间对齐
                if ir_re is not None:
                    ir_img = cv2.resize(ir_img, (704, 419), cv2.INTER_AREA)
                    ir_img = ir_img[:, ir_re[0]:ir_re[1]]
                    tm_img = tm_img[tm_re[0]:tm_re[1]]
                
                ir_img = img_convert(img=ir_img, W=self.W, H=self.H, is_single=False) # (H, W, 3) BGR
                tm_img = img_convert(img=tm_img, W=self.W, H=self.H, is_single=True) # (1, 1, H, W) BGR
                Y, Cr, Cb = BGR2YCrCb(ir_img) # (1, 1, H, W), (H, W), (H, W) BGR

                inputs = [Y, tm_img] # [(1, 1, H, W), (1, 1, H, W)] [BGR, BGR]
                self.preprocess_queue.put((idx, (inputs, Cr, Cb))) # [(1, 1, H, W), (1, 1, H, W)] [BGR, BGR]
                idx += 1

                ir_frame_id += 1
                tm_frame_id += 1
                ir_next_time += frame_interval
                tm_next_time += frame_interval
            else:
                # 只推进时间戳更小的一方
                if ir_time < tm_time:
                    cap_ir.read()  # 跳过多余帧
                    ir_frame_id += 1
                else:
                    cap_tm.read()  # 跳过多余帧
                    tm_frame_id += 1

        # 释放资源
        cap_ir.release()
        cap_tm.release()

        # 通知融合线程数据传输完毕
        for _ in range(self.fuse_thread_count):
            self.preprocess_queue.put((self.FINAL, self.SIGNAL))

    def fuse(self):
        # 加载模型
        fusemodel = load_model(
            model_path=self.fuse_model_path,
            target=self.target,
        )

        while True:
            # 外部中断
            if self.stop_event.is_set():
                break

            idx, item = self.preprocess_queue.get()
            if item is self.SIGNAL:
                # 得到数据传输完毕的通知，关闭线程，并通知检测线程有一个融合线程已经处理完毕
                self.fuse_queue.put((self.FINAL, self.SIGNAL))
                self.preprocess_queue.task_done()
                break

            inputs, Cr, Cb = item # [(1, 1, H, W), (1, 1, H, W)] [BGR, BGR]
            fuse_img = fusemodel.inference(inputs=inputs, data_format="nchw")[0] # (1, 1, H, W) BGR

            self.fuse_queue.put((idx, (fuse_img, Cr, Cb)))
            self.preprocess_queue.task_done()
 
        # 释放资源
        fusemodel.release()

    def detection(self):
        # 加载模型
        yolomodel = load_model(
            model_path=self.yolo_model_path,
            target=self.target,
        )

        while True:
            # 外部中断
            if self.stop_event.is_set():
                break

            # 所有融合线程都处理完毕
            if self.fuse_finished_event.is_set():
                self.detection_queue.put((self.FINAL, self.SIGNAL))
                break
            try:
                idx, item = self.fuse_queue.get(timeout=0.5)
                if item is self.SIGNAL:
                    with self.fuse_finished_lock:  # 临界资源互斥锁
                        self.fuse_finished_count += 1
                        self.fuse_queue.task_done()
                        # 所有的融合线程已经处理完毕，则标记
                        if self.fuse_finished_count == self.fuse_thread_count:
                            self.fuse_finished_event.set()
                            self.detection_queue.put((self.FINAL, self.SIGNAL))
                            break
                        continue

                fuse_img, Cr, Cb = item # (1, 1, H, W), (H, W), (H, W) BGR

                fuse_img = YCrCb2RGB(Y=fuse_img, Cr=Cr, Cb=Cb)  # (H, W, 3) RGB
                fuse_img = np.clip(fuse_img * 255.0, 0, 255).astype(np.uint8)  # (H, W, 3) RGB
                fuse_img = cv2.resize(fuse_img, (self.IMG_SIZE, self.IMG_SIZE), cv2.INTER_LANCZOS4)  # (IMG_SIZE, IMG_SIZE, 3) RGB
                fuse_img = np.transpose(fuse_img, (2, 0, 1)) # (3, IMG_SIZE, IMG_SIZE) RGB

                fuse_img_process = np.expand_dims(fuse_img, 0) # (1, 3, IMG_SIZE, IMG_SIZE) RGB

                output = yolomodel.inference(inputs=[fuse_img_process], data_format=["nchw"]) # (1, 1, 5, 8400) RGB

                self.detection_queue.put((idx, (output[0][0], fuse_img)))
                self.fuse_queue.task_done()

            except Empty:
                continue

        # 释放资源
        yolomodel.release()            

    def postprocess(self):
        expected_index = 0
        buffer = {}

        while True:
            # 外部中断
            if self.stop_event.is_set():
                break

            # 所有检测线程都处理完毕
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
                            break
                        continue

                else:
                    # 将图像放入缓冲池
                    buffer[idx] = item
                    while expected_index in buffer:
                        output, fuse_img = buffer.pop(expected_index) # (5, 8400) uint8

                        # 画框
                        num, fuse_img = draw(
                            img=fuse_img,
                            W=self.W,
                            H=self.H, 
                            IMG_SIZE=self.IMG_SIZE,
                            output=output,
                            conf_threshold=self.conf_threshold,
                        )

                        self.postprocess_queue.put((num, fuse_img))
                        expected_index += 1
                        self.detection_queue.task_done()

            except Empty:
                continue

    def reset(self):
        # 完成的线程数
        self.fuse_finished_count = 0
        self.detect_finished_count = 0
        
        # 临界资源互斥锁
        self.fuse_finished_lock = Lock()
        self.detect_finished_lock = Lock()
        
        # 线程完成事件
        self.fuse_finished_event = Event()
        self.detect_finished_event = Event()
        
        # 同步队列
        self.preprocess_queue = Queue(maxsize=self.maxsize)
        self.fuse_queue = Queue()
        self.detection_queue = Queue()
        self.postprocess_queue = Queue()

    def release(self):
        # 外部中断清理
        with self.preprocess_queue.mutex:
            self.preprocess_queue.queue.clear()
        with self.fuse_queue.mutex:
            self.fuse_queue.queue.clear()
        with self.detection_queue.mutex:
            self.detection_queue.queue.clear()
        with self.postprocess_queue.mutex:
            self.postprocess_queue.queue.clear()

    def run(self):
        # 重置参数
        self.reset()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 图像预处理
            pool.submit(self.preprocess)

            # 图像融合
            for _ in range(self.fuse_thread_count):
                pool.submit(self.fuse)

            # 人体检测
            for _ in range(self.detect_thread_count):
                pool.submit(self.detection)

            # 图像输出
            pool.submit(self.postprocess)

            # 等待任务完成
            self.preprocess_queue.join()
            self.fuse_queue.join()
            self.detection_queue.join()
            self.postprocess_queue.join()

            # 释放资源
            self.release()