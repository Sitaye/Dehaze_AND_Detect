import cv2
import numpy as np
from rknn.api import RKNN

# 图像大小
H = 480
W = 854

def load_model(model_path: str, target: str='rk3588', core_mask: int=RKNN.NPU_CORE_AUTO, do_perf_debug=False, do_eval_mem=False) -> RKNN:
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

def img_read(path: str, is_single=False, is_int8=False) -> np.ndarray:
    """
    返回单通道图像或者三通道图像

    Args:
        path (str): 图像路径
        is_single (bool): 是否为单通道图像
        is_int8 (bool): 是否保持图像为 int8 类型

    Returns:
        img (np.ndarray): 处理后的图像
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


def BGR2YCrCb(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 BGR 图像转换为 YCrCb 色彩空间，并分离通道。

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


def YCrCb2RGB(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """
    将 YCrCb 通道合并并转换为 RGB 图像。

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


QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(input[..., 2:4]*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    # print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
    # print('-' * 50)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[cl], score, top, left, right, bottom))

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
