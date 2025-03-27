import cv2
import numpy as np
from rknn.api import RKNN

# 图像大小
H = 540
W = 960

def load_model(model_path: str, target: str='rk3588', core_mask: int=RKNN.NPU_CORE_0_1_2, do_perf_debug=False, do_eval_mem=False) -> RKNN:
    """
    加载模型
    
    Args:
        model_path (str): 目标模型文件的位置
        target (str): 运行的硬件平台
        core_mask (int): NPU 核心掩码
        do_perf_debug (bool): 是否进行性能分析，调用 `rknn.val_perf()` 接口可以获取模型运行的总时间
        do_eval_mem (bool): 是否进行内存评估模式，调用 `rknn.eval_memory()` 接口获取模型运行时的内存使用情况

    Returns:
        RKNN: 返回模型
    """
    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(
        target=target,
        perf_debug=do_perf_debug,
        core_mask=core_mask,
        eval_mem=do_eval_mem,
    )
    return rknn

def img_read(path: str, is_single=False, is_int8=False) -> np.ndarray:
    """
    返回单通道图像或者三通道图像

    Args:
        path (str): 图像路径
        is_single (bool): 是否为单通道图像
        is_int8 (bool): 是否保持图像为 int8 类型

    Returns:
        np.ndarray: 处理后的图像
    """
    if is_single:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
    else:
        img = cv2.imread(path)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)

    if not is_int8:
        img = img.astype(np.float32) / 255.0
            
    return img


def img_save(img: np.ndarray, save_path: str, is_BGR=True, is_int8=True):
    """
    保存图像到指定路径。

    Args:
        img (np.ndarray): 输入图像
        save_path (str): 保存路径
        is_BGR (bool): 是否保存为 BGR 格式
        is_int8 (bool): 是否将图像保存为 uint8 类型
    """
    if len(img.shape) == 4 and img.shape[0] == 1 and img.shape[1] == 1:
        img = np.squeeze(img, axis=(0, 1))  # (1, 1, H, W) -> (H, W)
    elif len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # (3, H, W) -> (H, W, 3)

    if is_int8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if not is_BGR:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(save_path, img)


def BGR2YCrCb(img: np.ndarray):
    """
    将 BGR 图像转换为 YCrCb 色彩空间，并分离通道。

    Args:
        img (np.ndarray): 输入的 BGR 图像，(H, W, 3)

    Returns:
        Y (np.ndarray): 亮度通道，(1, 1, H, W)
        Cr (np.ndarray): 红色色度偏移通道，(H, W)
        Cb (np.ndarray): 蓝色色度偏移通道，(H, W)
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    Y = np.expand_dims(np.expand_dims(Y, axis=0), axis=0)

    return Y, Cr, Cb


def YCrCb2BGR(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """
    将 YCrCb 通道合并并转换为 BGR 图像。

    Args:
        Y (np.ndarray): 亮度通道，(1, 1, H, W)
        Cr (np.ndarray): 红色色度偏移通道，(H, W)
        Cb (np.ndarray): 蓝色色度偏移通道，(H, W)

    Returns:
        img (np.ndarray): 转换后的 BGR 图像，形状为 (H, W, 3)
    """
    Y = np.squeeze(np.squeeze(Y, axis=0), axis=0)

    img = cv2.merge([Y, Cr, Cb])
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    return img