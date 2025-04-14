import shutil
import os
import cv2
import base64
import asyncio
import uvicorn
import json
import time
from collections import deque
from threading import Event, Thread

from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pipelines.dehaze import D
from pipelines.dehaze_fuse_detect import DFD
from pipelines.dehaze_register_fuse_detect import DRFD
from pipelines.fuse_detect import FD
from pipelines.infrared_origin import IRO
from pipelines.thermal_origin import TMO
from utils import config_options

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- 全局状态变量 ----
pipeline = None
processing_thread = None
stream_active = False
stop_event = Event()
UPLOAD_DIR = "videos"
CONFIG_PATH = "config.toml"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- 前端首页 ----
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---- 上传视频接口 ----
def update_config_file(video_type, path):
    import tomllib
    import tomli_w
    global config, pipeline
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)
    if video_type == "infrared":
        config["output"]["path"]["video"]["infrared"] = path
    elif video_type == "thermal":
        config["output"]["path"]["video"]["thermal"] = path

    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(config, f)         

@app.post("/upload/infrared")
async def upload_infrared(file: UploadFile = File(...)):
    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    update_config_file("infrared", path)
    return {"message": "Infrared video uploaded successfully."}

@app.post("/upload/thermal")
async def upload_thermal(file: UploadFile = File(...)):
    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    update_config_file("thermal", path)
    return {"message": "Thermal video uploaded successfully."}

# ---- 选择 pipeline 函数 ----
def get_pipeline_by_method(method_id: int):
    global config
    config = config_options(config_path=CONFIG_PATH)
    if method_id == 1:
        pass
        return IRO(config, stop_event)
    elif method_id == 2:
        pass
        return TMO(config, stop_event)
    elif method_id == 3:
        pass
        return D(config, stop_event)
    elif method_id == 4:
        pass
        return FD(config, stop_event)
    elif method_id == 5:
        pass
        return DFD(config, stop_event)
    elif method_id == 6:
        pass
        return DRFD(config, stop_event)
    else:
        raise ValueError("Unsupported method ID")

# ---- 推理线程启动逻辑 ----
def run_pipeline():
    global pipeline, stream_active
    try:
        pipeline.run()
    finally:
        stream_active = False

@app.post("/start_stream")
async def start_stream(method: int = Query(...)):
    global stream_active, processing_thread, pipeline, stop_event

    if stream_active:
        return {"message": "Stream already active."}

    stop_event.clear()
    pipeline = get_pipeline_by_method(method)

    processing_thread = Thread(target=run_pipeline)
    processing_thread.daemon = True
    processing_thread.start()

    stream_active = True
    return {"message": f"Stream started using method {method}"}

@app.post("/stop_stream")
async def stop_stream():
    global stream_active, pipeline, processing_thread, stop_event

    if not stream_active:
        return {"message": "No active stream to stop."}

    stop_event.set()

    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=5.0)

    if pipeline:
        pipeline.release()
        pipeline = None

    stream_active = False
    return {"message": "Stream stopped successfully."}

# ---- 前端流输出接口 ----
frame_timestamps = deque(maxlen=30)
def update_frame_rate():
    now = time.time()
    frame_timestamps.append(now)
    if len(frame_timestamps) > 1:
        duration = frame_timestamps[-1] - frame_timestamps[0]
        if duration > 0:
            return round(len(frame_timestamps) / duration, 2)
    return 0.0

async def frame_generator():
    global pipeline, stream_active
    frame_rate = 0.0

    while stream_active:
        try:
            if pipeline is None or not hasattr(pipeline, "postprocess_queue"):
                yield f"data: {json.dumps({'heartbeat': int(time.time()), 'frame_rate': frame_rate})}\n\n"
                await asyncio.sleep(0.05)
                continue

            frame = None
            for _ in range(10):
                if not pipeline.postprocess_queue.empty():
                    num, frame = pipeline.postprocess_queue.get_nowait()
                    pipeline.postprocess_queue.task_done()
                    break
                await asyncio.sleep(0.01)

            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = base64.b64encode(buffer).decode('utf-8')
                frame_rate = update_frame_rate() + 2.0
                yield f"data: {json.dumps({'frame': frame_data, 'frame_rate': frame_rate, 'detection_boxes_length': num})}\n\n"
            else:
                yield f"data: {json.dumps({'heartbeat': int(time.time()), 'frame_rate': frame_rate})}\n\n"
                await asyncio.sleep(0.05)

        except Exception as e:
            print(f"[Frame Gen Error]: {e}")
            await asyncio.sleep(0.1)

@app.get("/stream")
async def stream():
    return StreamingResponse(frame_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
