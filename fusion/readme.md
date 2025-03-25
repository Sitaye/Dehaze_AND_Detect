
模型推理：
```
python infer2.py
    --vi_path <可见光图像文件夹>
    --ir_path <红外热成像图像文件夹>
    --res_path <融合图像结果保存文件夹>
    --mode_path <RKNN文件路径>
```

模型量化
```
python infer.py
    --dtype <量化精度>
    --metod <量化方法 channel or layer>
    --algorithm <量化算法 normal or mmse or kl>
    --fuse_input_list <输入数据格式>
    --qua <是否做量化>
    --datasets_file <组织量化数据的txt文件>
    --export_model_name <导出文件名>

```

