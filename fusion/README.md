### 模型量化
```shell
python quantize.py
    --dtype {w8a8,w8a16,w16a16i,w16a16i_dfp} 量化精度
    --algorithm {normal,mmse,kl_divergence}  量化算法
    --method {channel,layer}                 量化方法
    --target {rk3588,...}                    硬件平台
    --do_pruning {True,False}                是否剪枝
    --model_path MODEL_PATH                  源模型路径
    --input_format INPUT_FORMAT              数据输入格式
    --do_quatize {True,False}                是否量化
    --dataset_file DATASET_FILE              量化数据集组织文件
    --batch_size BATCH_SIZE                  模型批次
    --export_model_name EXPORT_MODEL_NAME    模型导出路径
```

### 模型推理
```shell
python infer.py
    --model_path MODEL_PATH       模型文件路径
    --target {rk3588,...}         硬件平台
    --rknn_batch RKNN_BATCH      模型批次
    --infrared_path INFRARED_PATH 红外图像路径
    --thermal_path THERMAL_PATH   热成像图像路径
    --results_path RESULTS_PATH   融合结果保存路径
```

