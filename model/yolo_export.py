from ultralytics import YOLO

model = YOLO('model/origin/yolo11n_f16_aug2_nir.pt')
model.export(format="onnx")