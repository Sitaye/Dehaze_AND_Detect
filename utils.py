import cv2
import numpy as np
import SimpleITK as sitk
from rknn.api import RKNN

def config_options(config_path: str) -> dict:
    """
    读取 toml 配置文件

    Args:
        config_path (str): 配置文件所在路径
    
    Returns:
        config (dict): 配置文件转化为字典
    """
    import tomllib
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError as e:
        raise e
    except tomllib.TOMLDecodeError as e:
        raise e
    except Exception as e:
        raise e

    return config


def load_model(
    model_path: str,
    target: str='rk3588',
    core_mask: int=RKNN.NPU_CORE_AUTO,
    do_perf_debug: bool=False,
    do_eval_mem: bool=False
) -> RKNN:
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
    rknn = RKNN(verbose=True)
    rknn.load_rknn(model_path)
    rknn.init_runtime(
        target=target,
        perf_debug=do_perf_debug,
        core_mask=core_mask,
        eval_mem=do_eval_mem,
    )
    return rknn

def img_convert(img: np.ndarray, W: int, H: int, is_single: bool=False, is_int8: bool=False) -> np.ndarray:
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

def register_init(
    numberOfHistogramBins: int=50, 
    learningRate: float=1.0,
    minStep: float=1e-4,
    numberOfIterations: int=500,
    SamplingStrategy: float=0.5,
) -> sitk.ImageRegistrationMethod:
    """
    配准方法初始化

    Args:
        numberOfHistogramBins (int): 用于 Mattes 互信息度量的直方图的 bin 数量
        learningRate (float): 学习率
        minStep (float): 最小步长
        numberOfIterations (int): 最大迭代次数
        SamplingStrategy (float): 采样比例

    Returns:
        registration_method (sitk.ImageRegistrationMethod): 初始化的配准方法
    """
    registration_method = sitk.ImageRegistrationMethod()
    
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=numberOfHistogramBins)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learningRate, minStep=minStep, numberOfIterations=numberOfIterations
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetMetricSamplingPercentage(SamplingStrategy)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)

    return registration_method

def img_register(
    ir_img: np.ndarray,
    tm_img: np.ndarray,
    registration_method: sitk.ImageRegistrationMethod,
    transform: sitk.Transform,
    is_key: bool=False,
) -> tuple[np.ndarray, np.ndarray, sitk.Transform]:
    """
    图像配准

    Args:
        ir_img (np.ndarray): 红外图像
        tm_img (np.ndarray): 热成像图像
        registration_method (sitk.ImageRegistrationMethod): 配准方法
        transform (sitk.Transform): 转换参数
        is_key (bool): 是否为关键帧

    Returns:
        tuple[np.ndarray,np.ndarray,sitk.Transform]: 
            - ir_img (np.ndarray): 配准后的红外图像
            - tm_img (np.ndarray): 配准后的热成像图像
            - final_transform (sitk.Transform): 更新后的转换参数
    """
    ir_img_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)

    ir_img_sitk = sitk.GetImageFromArray(ir_img_gray.astype(np.float32))
    tm_img_sitk = sitk.GetImageFromArray(tm_img.astype(np.float32))    

    if not transform:
        initial_transform = sitk.CenteredTransformInitializer(
            ir_img_sitk, tm_img_sitk, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
    else:
        registration_method.SetInitialTransform(transform, inPlace=False)

    if is_key:
        final_transform = registration_method.Execute(ir_img_sitk, tm_img_sitk)
    else:
        final_transform = transform

    resampler = sitk.ResampleImageFilter()
    # 降采样减少计算量
    target_size = (320, 180)
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing([
        ir_img_sitk.GetSpacing()[0] * (ir_img_sitk.GetSize()[0] / target_size[0]),
        ir_img_sitk.GetSpacing()[1] * (ir_img_sitk.GetSize()[1] / target_size[1]),
    ])
    resampler.SetReferenceImage(ir_img_sitk)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    tm_img_transformed = resampler.Execute(tm_img_sitk)

    tm_img = sitk.GetArrayFromImage(tm_img_transformed)
    tm_img = cv2.normalize(tm_img, None, 0, 255, cv2.NORM_MINMAX)
    tm_img = np.expand_dims(np.expand_dims(tm_img, axis=0), axis=0)
    
    ir_img = ir_img.astype(np.float32) / 255.0
    tm_img = tm_img.astype(np.float32) / 255.0
    
    return ir_img, tm_img, final_transform

def BGR2YCrCb(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 BGR 图像转换为 YCrCb 色彩空间，并分离通道

    Args:
        img (np.ndarray): 输入的 BGR 图像，(H, W, 3)

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]:
            - Y (np.ndarray): Y 亮度通道，(1, 1, H, W)
            - Cr (np.ndarray): Cr 红色色度偏移通道，(H, W)
            - Cb (np.ndarray): Cb 蓝色色度偏移通道，(H, W)
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

def draw(
    img: np.ndarray,
    W: int,
    H: int,
    IMG_SIZE: int,
    output: tuple,
    conf_threshold: float,
) -> tuple[int, np.ndarray]:
    """
    图像识别画框

    Args:
        img (np.ndarray): 原始图像
        W (int): 图像的宽度
        H (int): 图像的高度
        output (tuple): 模型输出的识别信息
        conf_threshold (float): 置信阈值

    Returns:
        tuple[int, np.ndarray]:
            - num: 识别框的个数
            - img: 画框之后的图像
    """
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
    
    img = np.transpose(img, (1, 2, 0)) # (IMG_SIZE, IMG_SIZE, 3)
    img = cv2.resize(img, (W, H), cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if len(indices) > 0:
        for i in indices.flatten():
            x_min_i, y_min_i, x_max_i, y_max_i = boxes[i]

            color = (3, 247, 3)
            cv2.rectangle(img, (x_min_i, y_min_i), (x_max_i, y_max_i), color, 2)

            label = f"Person: {confidences[i]:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            text_top = y_min_i - text_height - baseline
            text_bottom = y_min_i

            if text_top < 0:
                text_top = 0
                text_bottom = text_height + baseline

            if text_bottom > img.shape[0]:
                text_top = img.shape[0] - (text_height + baseline)
                text_bottom = img.shape[0]

            cv2.rectangle(img, (x_min_i - 1, text_top), (x_min_i + text_width, text_bottom), color, -1)
            cv2.putText(img, label, (x_min_i, text_bottom - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        return len(indices), img
    else:
        return 0, img

def correct_output_image(
    img: np.ndarray,
    base_gamma: float=1.2,
    brightness_boost: int=30,
    use_clahe: bool=True,
) -> np.ndarray:
    """
    红外图像结构增强

    Args:
        img (np.ndarray): 原始红外图像
        base_gamma (float): 伽马值，控制图像的对比度
        brightness_boost (int): 亮度提升值
        use_clahe (bool): 是否使用 CLAHE

    Returns:
        img (np.ndarray): 增强之后的红外图像
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_br, a, b = cv2.split(lab)
    avg_brightness = np.mean(l_br)

    if avg_brightness < 150:
        gamma = base_gamma * 1.2
        brightness_boost += 20
    else:
        gamma = base_gamma
        brightness_boost = max(0, brightness_boost - 20)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_br = clahe.apply(l_br)
        l_br = cv2.add(l_br, brightness_boost)
        l_br = np.clip(l_br, 0, 255)
        lab = cv2.merge((l_br, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = np.power(img / 255.0, gamma)
    img = np.uint8(np.clip(img * 255, 0, 255))

    return img

def preprocess_image(img, W, H):
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # [3, H, W]
    img = np.expand_dims(img, axis=0)  # [1, 3, H, W]
    return img

def postprocess_output(output_tensor):
    img = output_tensor[0]  # [3, H, W]
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
    img = np.uint8(np.clip(img * 255.0, 0, 255))
    return img
