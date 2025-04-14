import cv2
import numpy as np
from queue import Empty, Queue
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor

from utils import correct_output_image, preprocess_image, postprocess_output, load_model


class D:
    def __init__(self, config, stop_event=Event()):
        # 常量参数
        self.H = config["output"]["H"]
        self.W = config["output"]["W"]
        self.SIGNAL = object()
        self.FINAL = float("inf")

        # 数据参数
        self.infrared_video_path = config["output"]["path"]["video"]["infrared"]
        
        # RKNN 参数
        self.target = config["rknn"]["target"]
        self.dehaze_model_path = config["rknn"]["model"]["dehaze"]["path"]

        # 线程参数
        self.stop_event = stop_event
        self.maxsize = config["thread"]["maxsize"]
        self.max_workers = config["thread"]["max_workers"]
        self.dehaze_thread_count = 12

    def preprocess(self):
        # 获取视频流
        cap_ir = cv2.VideoCapture(self.infrared_video_path)

        idx = 0
        while cap_ir.isOpened():
            # 外部中断
            if self.stop_event.is_set():
                break

            # 读取视频帧
            ret_ir, ir_img = cap_ir.read()
            if not ret_ir:
                break

            ir_img = cv2.resize(ir_img, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4) # (H, W, 3) BGR
            lab = cv2.cvtColor(ir_img, cv2.COLOR_BGR2LAB)
            l_channel = cv2.split(lab)[0]
            avg_brightness = np.mean(l_channel)
            enhanced_img = correct_output_image(ir_img) # (H, W, 3) RGB

            self.preprocess_queue.put((idx, (enhanced_img, avg_brightness))) # (H, W, 3) BGR
            idx += 1

        cap_ir.release()

        # 通知去烟线程数据传输完毕
        for _ in range(self.dehaze_thread_count):
            self.preprocess_queue.put((self.FINAL, self.SIGNAL))

    def dehaze(self):
        # 加载模型
        dehazemodel = load_model(
            model_path=self.dehaze_model_path,
            target=self.target,
        )

        while True:
            # 外部中断
            if self.stop_event.is_set():
                break

            idx, item = self.preprocess_queue.get()
            if item is self.SIGNAL:
                # 得到数据传输完毕的通知，关闭线程，并通知后处理线程有一个融合线程已经处理完毕
                self.dehaze_queue.put((self.FINAL, self.SIGNAL))
                self.preprocess_queue.task_done()
                break

            enhanced_img, avg_brightness = item # (H, W, 3) RGB

            if avg_brightness > 150:
                # 如果场景亮度过大则不应用 CNN 的去烟方法，只使用 CV 去烟
                dehaze_img = enhanced_img
            else:
                input_data = preprocess_image(enhanced_img, W=self.W, H=self.H)
                outputs = dehazemodel.inference(inputs=[input_data], data_format=["nchw"])
                dehaze_img = postprocess_output(outputs[0])
                dehaze_img = cv2.cvtColor(dehaze_img, cv2.COLOR_RGB2BGR) # (H, W, 3) BGR

            self.dehaze_queue.put((idx, (dehaze_img))) # (H, W, 3) BGR
            self.preprocess_queue.task_done()
            
        # 释放资源
        dehazemodel.release()
            
    def postprocess(self):
        expected_index = 0
        buffer = {}

        while True:
            # 外部中断
            if self.stop_event.is_set():
                break

            # 所有检测线程都处理完毕
            if self.dehaze_finished_event.is_set():
                break
            try:
                idx, item = self.dehaze_queue.get(timeout=0.5)
                if item is self.SIGNAL:
                    with self.dehaze_finished_lock:  # 临界资源互斥锁
                        self.dehaze_finished_count += 1
                        self.dehaze_queue.task_done()
                        # 所有的去烟线程已经处理完毕，则标记
                        if self.dehaze_finished_count == self.dehaze_thread_count:
                            self.dehaze_finished_event.set()
                            break
                        continue

                else:
                    # 将图像放入缓冲池
                    buffer[idx] = item
                    while expected_index in buffer:
                        dehaze_img = buffer.pop(expected_index) # (5, 8400) uint8
                        self.postprocess_queue.put((0, dehaze_img))
                        expected_index += 1
                        self.dehaze_queue.task_done()

            except Empty:
                continue

    def reset(self):
        # 临界资源互斥
        self.dehaze_finished_count = 0
        self.dehaze_finished_lock = Lock()
        self.dehaze_finished_event = Event()
        
        # 同步队列
        self.preprocess_queue = Queue(maxsize=self.maxsize)
        self.dehaze_queue = Queue()
        self.postprocess_queue = Queue()

    def release(self):
        # 外部中断清理
        with self.preprocess_queue.mutex:
            self.preprocess_queue.queue.clear()
        with self.dehaze_queue.mutex:
            self.dehaze_queue.queue.clear()
        with self.postprocess_queue.mutex:
            self.postprocess_queue.queue.clear()

    def run(self):
        # 重置参数
        self.reset()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 图像预处理
            pool.submit(self.preprocess)

            # 图像去烟
            for _ in range(self.dehaze_thread_count):
                pool.submit(self.dehaze)

            # 图像输出
            pool.submit(self.postprocess)

            # 等待任务完成
            self.preprocess_queue.join()
            self.dehaze_queue.join()
            self.postprocess_queue.join()

            # 释放资源
            self.release()