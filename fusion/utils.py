import cv2
import numpy as np
from rknn.api import RKNN

def load_model(model_path: str, target: str='rk3588', core_mask: int=RKNN.NPU_CORE_AUTO, do_perf_debug: bool=False, do_eval_mem: bool=False) -> RKNN:
    """
    加载模型
    
    Args:
        model_path (str): 目标模型文件的位置
        target (str): 运行的硬件平台
        core_mask (int): NPU 核心掩码
        do_perf_debug (bool): 是否进行性能分析，调用 `rknn.val_perf()` 接口可以获取模型运行的总时间
        do_eval_mem (bool): 是否进行内存评估模式，调用 `rknn.eval_memory()` 接口获取模型运行时的内存使用情况

    Returns:
        rknn (RKNN): 返回模型
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

def img_read(path: str, W: int, H: int, is_single=False, is_int8=False) -> np.ndarray:
    """
    读取图像

    Args:
        path (str): 图像路径
        W (int): 图片的宽
        H (int): 图片的高
        is_single (bool): 是否为单通道图像
        is_int8 (bool): 是否保持图像为 int8 类型

    Returns:
        img (np.ndarray): 返回处理后的图像
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


def img_save(img: np.ndarray, save_path: str, is_BGR=True, is_int8=True) -> None:
    """
    保存图像到指定路径

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

def img_convert(img: np.ndarray, W: int, H: int, is_single=False, is_int8=False) -> np.ndarray:
    """
    转化图像

    Args:
        img (np.ndarray): 输入图像
        W (int): 图片的宽
        H (int): 图片的高
        is_single (bool): 是否为单通道图像
        is_int8 (bool): 是否保持图像为 int8 类型

    Returns:
        img (np.ndarray): 返回处理后的图像
    """
    if is_single:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
    else:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)

    if not is_int8:
        img = img.astype(np.float32) / 255.0
            
    return img

def BGR2YCrCb(img: np.ndarray) -> tuple[np.ndarray]:
    """
    将 BGR 图像转换为 YCrCb 色彩空间，并分离通道

    Args:
        img (np.ndarray): 输入的 BGR 图像，(H, W, 3)

    Returns:
        (Y, Cr, Cb) (tuple(np.ndarray)): Y 亮度通道，(1, 1, H, W)； Cr 红色色度偏移通道，(H, W)；Cb 蓝色色度偏移通道，(H, W)
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    Y = np.expand_dims(np.expand_dims(Y, axis=0), axis=0)

    return Y, Cr, Cb

def YCrCb2BGR(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """
    将 YCrCb 通道合并并转换为 BGR 图像

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

def YCrCb2RGB(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """
    将 YCrCb 通道合并并转换为 RGB 图像

    Args:
        Y (np.ndarray): 亮度通道，(1, 1, H, W)
        Cr (np.ndarray): 红色色度偏移通道，(H, W)
        Cb (np.ndarray): 蓝色色度偏移通道，(H, W)

    Returns:
        img (np.ndarray): 转换后的 RGB 图像，形状为 (H, W, 3)
    """
    Y = np.squeeze(np.squeeze(Y, axis=0), axis=0)

    img = cv2.merge([Y, Cr, Cb])
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)

    return img

def draw(fuse_img, W, H, IMG_SIZE, output, conf_threshold):
    x_center, y_center, width, height, confidence = output
    mask = confidence > conf_threshold
    x_center, y_center, width, height, confidence = x_center[mask], y_center[mask], width[mask], height[mask], confidence[mask]

    x_min = np.maximum(0, ((x_center - width / 2) * (W / IMG_SIZE)).astype(np.int32))
    y_min = np.maximum(0, ((y_center - height / 2) * (H / IMG_SIZE)).astype(np.int32))
    x_max = np.minimum(W, ((x_center + width / 2) * (W / IMG_SIZE)).astype(np.int32))
    y_max = np.minimum(H, ((y_center + height / 2) * (H / IMG_SIZE)).astype(np.int32))
    
    boxes = np.array([x_min, y_min, x_max, y_max]).T 
    confidences = confidence
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=conf_threshold, nms_threshold=0.4)
    
    fuse_img = np.transpose(fuse_img, (1, 2, 0)) # (IMG_SIZE, IMG_SIZE, C)
    fuse_img = cv2.resize(fuse_img, (W, H))
    fuse_img = cv2.cvtColor(fuse_img, cv2.COLOR_RGB2BGR)

    if len(indices) > 0:
        for i in indices.flatten():
            x_min_i, y_min_i, x_max_i, y_max_i = boxes[i]

            color = (3, 247, 3)
            cv2.rectangle(fuse_img, (x_min_i, y_min_i), (x_max_i, y_max_i), color, 2)

            label = f"Person: {confidences[i]:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            text_top = y_min_i - text_height - baseline
            text_bottom = y_min_i

            if text_top < 0:
                text_top = 0
                text_bottom = text_height + baseline  # 确保文本框不变形，位置调整为图片顶部

            if text_bottom > fuse_img.shape[0]:
                text_top = fuse_img.shape[0] - (text_height + baseline)
                text_bottom = fuse_img.shape[0]

            cv2.rectangle(fuse_img, (x_min_i - 1, text_top), (x_min_i + text_width, text_bottom), color, -1)
            cv2.putText(fuse_img, label, (x_min_i, text_bottom - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return fuse_img
