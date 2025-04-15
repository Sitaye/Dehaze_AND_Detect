# “双瞳探焰，雾中搜生”—基于深度学习的双模态图像去烟与融合的火场人体识别系统

本项目为大学生服务外包创新创业大赛选题【A25】浓烟环境人体目标判别【中电海康】的项目仓库。

## Overview

```shell
app/                                    # 项目根目录
├── app.py                              # 主程序
├── config.toml                         # 配置文件
├── models/                             # 模型文件 
│   ├── dehaze.rknn                     # 去烟模型文件
│   ├── fuse.rknn                       # 融合模型文件
│   └── yolo11n.rknn                    # 识别模型文件
├── pipelines/                          # 执行程序模块
│   ├── dehaze_fuse_detect.py           # 去烟 + 融合 + 检测
│   ├── dehaze.py                       # 去烟
│   ├── dehaze_register_fuse_detect.py  # 去烟 + 配准 + 融合 + 检测
│   ├── fuse_detect.py                  # 融合 + 检测
│   ├── infrared_origin.py              # 原始红外
│   └── thermal_origin.py               # 原始热成像
├── README.md                           # README 文件
├── requirements.txt                    # 依赖库
├── static/                             # 静态文件
│   ├── logo.png                        # 项目 Logo
│   ├── placeholder.jpg                 # 视频占位符
│   └── upload.svg                      # 「上传」 图标
├── templates/                          # 前端文件夹
│   └── index.html                      # 前端 HTML
├── utils.py                            # 工具类
└── videos/                             # 双模态视频
```

## Installation

### 1. Clone the Repository

```shell
git clone https://github.com/Sitaye/Dehaze_AND_Detect
cd Dehaze_AND_Detect
```

### 2. Install Dependencies

Ensure you have Python 3.10 or higher installed.

```shell
pip install -r requirements.txt
```

## Usage

```python
python app.py
```

You will see

```shell
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```