from ultralytics import YOLO

# Load a model
model = YOLO('yolov5nu.pt')  # load an official model

# Export the model
# model.export(format='onnx', batch=128, imgsz=640, simplify=True, optimize=True, nms=False, int8=True,
#              dynamic=True)

# model.export(format='edgetpu', int8=True, batch=64, imgsz=(1920, 1088))

model.export(format="ncnn", imgsz=640, half=True)
