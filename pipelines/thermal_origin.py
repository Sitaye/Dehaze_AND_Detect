from time import sleep
import cv2
from queue import Queue
from threading import Event

class TMO:
    def __init__(self, config, stop_event=Event()):
        # 常量参数
        self.H = config["output"]["H"]
        self.W = config["output"]["W"]

        # 数据参数
        self.thermal_video_path = config["output"]["path"]["video"]["thermal"]

        # 线程参数
        self.stop_event = stop_event
        self.max_workers = config["thread"]["max_workers"]

    def release(self):
        # 外部中断清理
        with self.postprocess_queue.mutex:
            self.postprocess_queue.queue.clear()

    def reset(self):
        # 同步队列
        self.postprocess_queue = Queue()

    def run(self):
        # 重置参数
        self.reset()

        # 获取视频流
        cap_tm = cv2.VideoCapture(self.thermal_video_path)

        while cap_tm.isOpened():
            # 外部中断
            if self.stop_event.is_set():
                break

            # 读取视频帧
            ret_tm, tm_img = cap_tm.read()
            if not ret_tm:
                break

            self.postprocess_queue.put((0, tm_img))
            sleep(1/25)

        # 释放资源
        self.release()