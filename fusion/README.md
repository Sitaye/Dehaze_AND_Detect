### 模型量化
```shell
python quantize.py
    --dtype DTYPE                         量化精度 "w8a8", "w8a16", "w16a16i", "w16a16i_dfp"
    --algorithm ALGORITHM                 量化算法 "normal", "mmse", "kl_divergence"
    --method METHOD                       量化方法 "channel", "layer"
    --target TARGET                       硬件平台 "rk3588", ...
    --model_path MODEL_PATH               待量化的模型路径
    --input_format INPUT_FORMAT           输入数据格式 "[[1, 1, H, W], [1, 1, H, W]]"
    --do_quatize DO_QUATIZE               是否做量化 "False", "True"
    --dataset_file DATASET_FILE           量化数据集组织文件
    --batch_size BATCH_SIZE               多批次量化
    --export_model_name EXPORT_MODEL_NAME 导出模型路径
```

### 模型推理
```shell
python infer.py
    --model_path MODEL_PATH       模型文件路径
    --target TARGET               硬件平台 "rk3588", ...
    --rknn_batch RKNN_BATCH       多批次
    --infrared_path INFRARED_PATH 红外图像路径
    --thermal_path THERMAL_PATH   热成像图像路径
    --results_path RESULTS_PATH   融合结果保存路径
```

